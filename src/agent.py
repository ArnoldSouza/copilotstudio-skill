# agent.py bkp

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from pathlib import Path
import re
import json
import asyncio
import logging
from typing import Dict, Optional, Any, Tuple, List, Callable

from dotenv import load_dotenv

from microsoft_agents.hosting.core import (
    Authorization,
    AgentApplication,
    TurnState,
    TurnContext,
    MemoryStorage,
)

from microsoft_agents.activity import load_configuration_from_env
from microsoft_agents.authentication.msal import MsalConnectionManager
from microsoft_agents.hosting.aiohttp import CloudAdapter

# === Genie (Databricks) ===
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dashboards import GenieAPI

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)
agents_sdk_config = load_configuration_from_env(os.environ)

VERSION = os.getenv("VERSION", "genie-enabled-1.2")
STORAGE = MemoryStorage()
CONNECTION_MANAGER = MsalConnectionManager(**agents_sdk_config)
ADAPTER = CloudAdapter(connection_manager=CONNECTION_MANAGER)
AUTHORIZATION = Authorization(STORAGE, CONNECTION_MANAGER, **agents_sdk_config)

logger = logging.getLogger("copilotstudioskill")
logging.basicConfig(level=logging.INFO)

AGENT_APP = AgentApplication[TurnState](
    storage=STORAGE, adapter=ADAPTER, authorization=AUTHORIZATION, **agents_sdk_config
)

# ------------------------------------------------------------------------------
# Genie client + state
# ------------------------------------------------------------------------------

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
DATABRICKS_SPACE_ID = os.getenv("DATABRICKS_SPACE_ID")

DBX_ENABLED = bool(DATABRICKS_HOST and DATABRICKS_TOKEN and DATABRICKS_SPACE_ID)

_workspace_client: Optional[WorkspaceClient] = None
_genie_api: Optional[GenieAPI] = None

if DBX_ENABLED:
    try:
        _workspace_client = WorkspaceClient(host=DATABRICKS_HOST, token=DATABRICKS_TOKEN)
        _genie_api = GenieAPI(_workspace_client.api_client)
        logger.info("✅ Genie (Databricks) inicializado.")
    except Exception as e:
        logger.exception("Falha ao inicializar Genie: %s", e)
        _workspace_client = None
        _genie_api = None
        DBX_ENABLED = False
else:
    logger.warning("⚠️ Variáveis do Genie ausentes. Integração desativada.")

# user_id -> conversation_id
_user_to_genie_conversation: Dict[str, str] = {}

# Limites de segurança para render no Teams/Playground
MAX_ROWS = int(os.getenv("GENIE_MAX_ROWS", "50"))              # máx. linhas
CALL_TIMEOUT_SECONDS = int(os.getenv("GENIE_TIMEOUT", "60"))   # timeout por chamada
MAX_RETRIES = int(os.getenv("GENIE_MAX_RETRIES", "3"))         # retries p/ chamadas
GENIE_MAX_CHARS = int(os.getenv("GENIE_MAX_CHARS", "12000"))   # máx. chars por atividade
GENIE_MAX_COLS = int(os.getenv("GENIE_MAX_COLS", "8"))         # máx. colunas exibidas
GENIE_MAX_CELL_CHARS = int(os.getenv("GENIE_MAX_CELL_CHARS", "200"))  # máx. chars por célula

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _safe_attachments(obj: Any) -> List[Any]:
    atts = getattr(obj, "attachments", None)
    return atts or []

async def _with_retry(func: Callable[[], Any], *, retries: int = MAX_RETRIES, base_delay: float = 0.8):
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return await asyncio.wait_for(func(), timeout=CALL_TIMEOUT_SECONDS)
        except Exception as e:
            last_exc = e
            await asyncio.sleep(base_delay * (2 ** attempt))
    raise last_exc or RuntimeError("Unexpected retry error")

def _truncate_text(s: str, limit: int) -> str:
    if s is None:
        return ""
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 1)] + "…"

def _chunk_text(text: str, limit: int) -> List[str]:
    """
    Quebra o texto em blocos <= limit tentando respeitar quebras de linha.
    """
    text = text or ""
    chunks: List[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut == -1 or cut < limit * 0.6:
            # não achou uma quebra boa; corta seco
            cut = limit
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        chunks.append(text)
    # remove vazios
    return [c for c in chunks if c]

async def _send_markdown(context: TurnContext, md: str):
    """
    Envia markdown em múltiplas mensagens para evitar 413 no conector.
    """
    parts = _chunk_text(md, GENIE_MAX_CHARS)
    total = len(parts)
    for idx, part in enumerate(parts, 1):
        suffix = f"\n\n_{idx}/{total}_" if total > 1 else ""
        await context.send_activity(part + suffix)

async def ask_genie(question: str, space_id: str, conversation_id: Optional[str] = None) -> Tuple[str, str]:
    """
    Inicia/continua conversa no Genie e retorna:
      - {"message": "..."}         quando texto
      - {"columns","data","query_description"} quando resultados tabulares
    """
    assert _genie_api is not None and _workspace_client is not None

    loop = asyncio.get_running_loop()

    try:
        # 1) criar/continuar mensagem
        async def _create():
            if conversation_id is None:
                msg = await loop.run_in_executor(None, _genie_api.start_conversation_and_wait, space_id, question)
            else:
                msg = await loop.run_in_executor(None, _genie_api.create_message_and_wait, space_id, conversation_id, question)
            return msg

        initial_message = await _with_retry(_create)
        conversation_id = initial_message.conversation_id

        # 2) pegar query_result se houver
        query_result = None
        if getattr(initial_message, "query_result", None) is not None:
            async def _get_qr():
                return await loop.run_in_executor(None, _genie_api.get_message_query_result,
                                                  space_id, initial_message.conversation_id, initial_message.id)
            query_result = await _with_retry(_get_qr)

        # 3) conteúdo/attachments
        async def _get_msg():
            return await loop.run_in_executor(None, _genie_api.get_message,
                                              space_id, initial_message.conversation_id, initial_message.id)
        message_content = await _with_retry(_get_msg)

        # 4) tabular?
        if query_result and getattr(query_result, "statement_response", None):
            async def _get_stmt():
                return await loop.run_in_executor(None, _workspace_client.statement_execution.get_statement,
                                                  query_result.statement_response.statement_id)
            results = await _with_retry(_get_stmt)

            query_description = ""
            for att in _safe_attachments(message_content):
                q = getattr(att, "query", None)
                if q and getattr(q, "description", None):
                    query_description = q.description or ""
                    break

            return json.dumps({
                "columns": results.manifest.schema.as_dict(),
                "data": results.result.as_dict(),
                "query_description": query_description
            }), conversation_id

        # 5) texto
        for att in _safe_attachments(message_content):
            t = getattr(att, "text", None)
            if t and getattr(t, "content", None):
                return json.dumps({"message": t.content}), conversation_id

        return json.dumps({"message": getattr(message_content, "content", "") or ""}), conversation_id

    except Exception as e:
        logger.exception("Erro no ask_genie: %s", e)
        return json.dumps({"error": "Falha ao processar sua solicitação no Genie."}), (conversation_id or "")

def _truncate_rows(rows: List[List[Any]], max_rows: int) -> Tuple[List[List[Any]], Optional[int]]:
    if not rows:
        return [], None
    if len(rows) <= max_rows:
        return rows, None
    return rows[:max_rows], len(rows) - max_rows

def _limit_cols(cols_meta: Dict, rows: List[List[Any]]) -> Tuple[List[Dict[str, Any]], List[List[Any]], Optional[int]]:
    """
    Restringe para no máximo GENIE_MAX_COLS colunas. Retorna (cols, rows, hidden_cols)
    """
    meta_cols = cols_meta.get("columns", []) or []
    if len(meta_cols) <= GENIE_MAX_COLS:
        return meta_cols, rows, None

    kept = meta_cols[:GENIE_MAX_COLS]
    hidden = len(meta_cols) - GENIE_MAX_COLS
    new_rows = [r[:GENIE_MAX_COLS] for r in rows]
    return kept, new_rows, hidden

def _fmt_cell(value: Any, type_name: str) -> str:
    t = (type_name or "").upper()
    if value is None:
        return "NULL"
    try:
        if t in ("DECIMAL", "DOUBLE", "FLOAT"):
            return _truncate_text(f"{float(value):,.2f}", GENIE_MAX_CELL_CHARS)
        if t in ("INT", "BIGINT", "LONG"):
            return _truncate_text(f"{int(value):,}", GENIE_MAX_CELL_CHARS)
        return _truncate_text(str(value), GENIE_MAX_CELL_CHARS)
    except Exception:
        return _truncate_text(str(value), GENIE_MAX_CELL_CHARS)

def format_genie_answer_md(answer_json: Dict) -> str:
    if "error" in answer_json:
        return f"⚠️ {answer_json['error']}"

    parts: List[str] = []

    if answer_json.get("query_description"):
        parts.append(f"**Query**: {answer_json['query_description']}")

    if "columns" in answer_json and "data" in answer_json:
        cols_meta = answer_json["columns"] or {}
        data = answer_json["data"] or {}

        raw_rows = data.get("data_array", []) or []
        rows, hidden_rows = _truncate_rows(raw_rows, MAX_ROWS)

        # limitar colunas
        meta_cols, rows, hidden_cols = _limit_cols(cols_meta, rows)

        if meta_cols:
            headers = [c.get("name", f"col{i+1}") for i, c in enumerate(meta_cols)]
            parts.append("| " + " | ".join(headers) + " |")
            parts.append("|" + "|".join(["---"] * len(headers)) + "|")

            for row in rows:
                formatted: List[str] = []
                for value, col in zip(row, meta_cols):
                    formatted.append(_fmt_cell(value, col.get("type_name") or ""))
                parts.append("| " + " | ".join(formatted) + " |")

            notes: List[str] = []
            if hidden_rows:
                notes.append(f"{hidden_rows} linha(s) ocultada(s)")
            if hidden_cols:
                notes.append(f"{hidden_cols} coluna(s) ocultada(s)")
            if notes:
                parts.append("\n_" + " • ".join(notes) + ". Refine sua pergunta para ver menos dados._")
        else:
            parts.append("_Sem colunas para exibir._")

    elif "message" in answer_json:
        parts.append(answer_json["message"] or "_Sem conteúdo._")

    else:
        parts.append("_Sem dados disponíveis._")

    return "\n".join(parts)

def _is_genie_command(text: str) -> bool:
    t = text.strip().lower()
    return (
        t.startswith("/genie ")
        or t.startswith("/g ")
        or t.startswith("genie ")
        or t in ("/genie", "genie", "/g")
    )

def _strip_genie_prefix(text: str) -> str:
    t = re.sub(r"^/(genie|g)\s*:?\s*", "", text, flags=re.IGNORECASE).strip()
    t = re.sub(r"^genie\s*:?\s*", "", t, flags=re.IGNORECASE).strip()
    return t

def _help() -> str:
    return (
        f"**Databricks Genie Help** (v{VERSION})\n"
        "- `/genie <pergunta>` ou `/g <pergunta>`: pergunta ao Genie\n"
        "- `/genie reset`: reinicia seu contexto de conversa com o Genie\n"
        "- Ex.: `/genie AP overdue by vendor in 2024`\n"
        f"- Limites: linhas={MAX_ROWS}, colunas={GENIE_MAX_COLS}, chars/atividade={GENIE_MAX_CHARS}, chars/célula={GENIE_MAX_CELL_CHARS}\n"
        f"- Timeout por chamada: {CALL_TIMEOUT_SECONDS}s\n"
    )

# ------------------------------------------------------------------------------
# Handlers
# ------------------------------------------------------------------------------

@AGENT_APP.conversation_update("membersAdded")
async def on_members_added(context: TurnContext, _state: TurnState):
    await context.send_activity(
        f"Olá! Echo + Genie habilitado. Versão: {VERSION}\n"
        f"• Use **/genie** ou **/g** para perguntar ao Databricks Genie.\n"
        f"• Ex.: `/genie Top 5 customers by revenue`\n"
        f"• Comando: `/genie reset` para reiniciar a conversa.\n"
        f"• `/genie help` para ver dicas."
    )

@AGENT_APP.activity("message")
async def on_message(context: TurnContext, _state: TurnState):
    text = (context.activity.text or "").strip()
    if not text:
        await context.send_activity("Envie uma mensagem para começar. 🙂")
        return

    lower = text.lower()

    # Info de versão
    if " version" in f" {lower} " or lower.strip() == "version":
        await context.send_activity(f"Running on version {VERSION}")

    # Ajuda
    if lower in ("/genie help", "genie help", "/g help"):
        await context.send_activity(_help())
        return

    # Reset
    if lower in ("/genie reset", "genie reset", "/g reset"):
        _user_to_genie_conversation.pop(context.activity.from_property.id, None)
        await context.send_activity("🔄 Conversa do Genie foi reiniciada para você.")
        return

    # Roteamento Genie
    if _is_genie_command(text):
        if not DBX_ENABLED:
            await context.send_activity(
                "⚠️ Genie não configurado. Defina `DATABRICKS_HOST`, `DATABRICKS_TOKEN` e `DATABRICKS_SPACE_ID` no .env."
            )
            return

        question = _strip_genie_prefix(text)
        if not question:
            await context.send_activity("Diga algo após `/genie ...` para perguntar ao Databricks Genie.")
            return

        user_id = context.activity.from_property.id
        conv_id = _user_to_genie_conversation.get(user_id)

        answer, new_conv = await ask_genie(question, DATABRICKS_SPACE_ID, conv_id)
        if new_conv:
            _user_to_genie_conversation[user_id] = new_conv

        try:
            parsed = json.loads(answer)
        except Exception:
            parsed = {"message": answer}

        md = format_genie_answer_md(parsed)
        await _send_markdown(context, md)   # <- envio com chunking para evitar 413
        return

    # Echo padrão
    await context.send_activity(f"Echo: {text}")
