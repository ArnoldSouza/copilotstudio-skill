# agent.py

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
# NOTE: Per user request, do not modify this section.

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)
agents_sdk_config = load_configuration_from_env(os.environ)

VERSION = os.getenv("VERSION", "databricks-genie-teams-1.0")
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
# NOTE: Per user request, do not modify this section.

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

# Safety limits for Teams/Playground render
MAX_ROWS = int(os.getenv("GENIE_MAX_ROWS", "50"))              # max rows
CALL_TIMEOUT_SECONDS = int(os.getenv("GENIE_TIMEOUT", "60"))   # per-call timeout
MAX_RETRIES = int(os.getenv("GENIE_MAX_RETRIES", "3"))         # retries for calls
GENIE_MAX_CHARS = int(os.getenv("GENIE_MAX_CHARS", "12000"))   # max chars per activity
GENIE_MAX_COLS = int(os.getenv("GENIE_MAX_COLS", "16"))        # max displayed columns (default 16)
GENIE_MAX_CELL_CHARS = int(os.getenv("GENIE_MAX_CELL_CHARS", "200"))  # max chars per cell

# ------------------------------------------------------------------------------
# User-configurable settings (chat overrides)
# ------------------------------------------------------------------------------

# In-memory map of user_id -> settings
_user_settings: Dict[str, Dict[str, int]] = {}

def _default_settings() -> Dict[str, int]:
    """
    Returns the default per-user limits. These values are used when a user
    has not set any overrides via the 'config' chat command.
    """
    return {
        "rows": MAX_ROWS,
        "cols": GENIE_MAX_COLS,
        "chars": GENIE_MAX_CHARS,
        "cell_chars": GENIE_MAX_CELL_CHARS,
        "timeout": CALL_TIMEOUT_SECONDS,
    }

def _get_settings_for(user_id: str) -> Dict[str, int]:
    """
    Gets the current settings for a user, initializing with defaults if needed.
    """
    s = _user_settings.get(user_id)
    if not s:
        s = _default_settings()
        _user_settings[user_id] = s
    return s

def _apply_overrides(user_id: str, overrides: Dict[str, int]) -> Dict[str, int]:
    """
    Applies the provided overrides to the user's settings and returns the result.
    """
    s = _get_settings_for(user_id).copy()
    s.update(overrides)
    _user_settings[user_id] = s
    return s

def _parse_config_overrides(text: str) -> Optional[Dict[str, int]]:
    """
    Parses a 'config' style message for overrides. Supports Portuguese and English keys.

    Examples:
      - 'config rows=100 cols=20 timeout=90'
      - 'config linhas=200 colunas=25'
      - 'set columns=12 chars=15000 cell=300'

    Returns:
      A dict with normalized keys: {'rows', 'cols', 'chars', 'cell_chars', 'timeout'}
      or None if nothing to update.
    """
    if not text:
        return None

    # Detect user intent to configure even if no pairs are present
    intent = bool(re.search(r'\b(config|settings?|set|ajuste|limites?)\b', text, flags=re.I))

    # Extract key=value pairs (supports PT/EN variants)
    pairs = re.findall(
        r'(?i)\b(rows|linhas|cols|colunas|columns|timeout|chars|cell|cell_chars|célula|celula)\s*[:=]\s*(\d+)',
        text
    )
    if not pairs and not intent:
        return None

    keymap = {
        "rows": "rows", "linhas": "rows",
        "cols": "cols", "colunas": "cols", "columns": "cols",
        "timeout": "timeout",
        "chars": "chars",
        "cell": "cell_chars", "cell_chars": "cell_chars", "célula": "cell_chars", "celula": "cell_chars",
    }

    out: Dict[str, int] = {}
    for k, v in pairs:
        nk = keymap.get(k.lower())
        if nk:
            try:
                out[nk] = max(1, int(v))
            except Exception:
                # Ignore non-integer or invalid values silently
                pass
    return out if out else ({} if intent else None)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _safe_attachments(obj: Any) -> List[Any]:
    """
    Safely returns a list of attachments from a message-like object.
    """
    atts = getattr(obj, "attachments", None)
    return atts or []

async def _with_retry(
    func: Callable[[], Any],
    *,
    retries: int = MAX_RETRIES,
    timeout: Optional[float] = None,
    base_delay: float = 0.8
):
    """
    Awaits a function with retry + timeout behavior.

    Args:
      func: async function (or wrapper) to execute.
      retries: number of retry attempts.
      timeout: per-attempt timeout (seconds). Falls back to CALL_TIMEOUT_SECONDS if None/<=0.
      base_delay: backoff base delay in seconds (exponential backoff).
    """
    last_exc: Optional[Exception] = None
    _timeout = timeout if timeout and timeout > 0 else CALL_TIMEOUT_SECONDS
    for attempt in range(retries):
        try:
            return await asyncio.wait_for(func(), timeout=_timeout)
        except Exception as e:
            last_exc = e
            # Exponential backoff: base_delay, 2*base_delay, 4*base_delay, ...
            await asyncio.sleep(base_delay * (2 ** attempt))
    raise last_exc or RuntimeError("Unexpected retry error")

def _truncate_text(s: str, limit: int) -> str:
    """
    Truncates a string to a given limit, appending ellipsis if needed.
    """
    if s is None:
        return ""
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 1)] + "…"

def _chunk_text(text: str, limit: int) -> List[str]:
    """
    Splits text into chunks of up to `limit` chars, attempting to break on newlines when possible.
    """
    text = text or ""
    chunks: List[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut == -1 or cut < limit * 0.6:
            # No natural break found; hard cut.
            cut = limit
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        chunks.append(text)
    # Remove empty segments
    return [c for c in chunks if c]

async def _send_markdown(context: TurnContext, md: str, *, max_chars: int):
    """
    Sends markdown in multiple messages to avoid connector 413 errors.

    Args:
      context: Bot framework turn context.
      md: Markdown content to send.
      max_chars: Max characters per activity (user-configurable).
    """
    parts = _chunk_text(md, max_chars)
    total = len(parts)
    for idx, part in enumerate(parts, 1):
        suffix = f"\n\n_{idx}/{total}_" if total > 1 else ""
        await context.send_activity(part + suffix)

async def ask_genie(
    question: str,
    space_id: str,
    conversation_id: Optional[str] = None,
    *,
    timeout: Optional[int] = None
) -> Tuple[str, str]:
    """
    Starts/continues a Genie conversation and returns a JSON string and conversation_id.

    Returns:
      - JSON with {"message": "..."} when it's a text answer
      - JSON with {"columns", "data", "query_description"} when tabular results are present
    """
    assert _genie_api is not None and _workspace_client is not None

    loop = asyncio.get_running_loop()

    try:
        # 1) Create or continue a message in the conversation
        async def _create():
            if conversation_id is None:
                return await loop.run_in_executor(None, _genie_api.start_conversation_and_wait, space_id, question)
            return await loop.run_in_executor(None, _genie_api.create_message_and_wait, space_id, conversation_id, question)

        initial_message = await _with_retry(_create, timeout=timeout)
        conversation_id = initial_message.conversation_id

        # 2) If there is a query_result, fetch it
        query_result = None
        if getattr(initial_message, "query_result", None) is not None:
            async def _get_qr():
                return await loop.run_in_executor(
                    None,
                    _genie_api.get_message_query_result,
                    space_id,
                    initial_message.conversation_id,
                    initial_message.id
                )
            query_result = await _with_retry(_get_qr, timeout=timeout)

        # 3) Fetch content/attachments for this message
        async def _get_msg():
            return await loop.run_in_executor(
                None,
                _genie_api.get_message,
                space_id,
                initial_message.conversation_id,
                initial_message.id
            )
        message_content = await _with_retry(_get_msg, timeout=timeout)

        # 4) Tabular result path
        if query_result and getattr(query_result, "statement_response", None):
            async def _get_stmt():
                return await loop.run_in_executor(
                    None,
                    _workspace_client.statement_execution.get_statement,
                    query_result.statement_response.statement_id
                )
            results = await _with_retry(_get_stmt, timeout=timeout)

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

        # 5) Text-only path
        for att in _safe_attachments(message_content):
            t = getattr(att, "text", None)
            if t and getattr(t, "content", None):
                return json.dumps({"message": t.content}), conversation_id

        # Fallback: raw content
        return json.dumps({"message": getattr(message_content, "content", "") or ""}), conversation_id

    except Exception as e:
        logger.exception("Error in ask_genie: %s", e)
        return json.dumps({"error": "Failed to process your request with Genie."}), (conversation_id or "")

def _truncate_rows(rows: List[List[Any]], max_rows: int) -> Tuple[List[List[Any]], Optional[int]]:
    """
    Truncates rows to a maximum of `max_rows`, returning (visible_rows, hidden_count).
    """
    if not rows:
        return [], None
    if len(rows) <= max_rows:
        return rows, None
    return rows[:max_rows], len(rows) - max_rows

def _limit_cols(
    cols_meta: Dict,
    rows: List[List[Any]],
    max_cols: int
) -> Tuple[List[Dict[str, Any]], List[List[Any]], Optional[int]]:
    """
    Limits columns to at most `max_cols`.

    Returns:
      (visible_cols_meta, adjusted_rows, hidden_cols_count)
    """
    meta_cols = cols_meta.get("columns", []) or []
    if len(meta_cols) <= max_cols:
        return meta_cols, rows, None

    kept = meta_cols[:max_cols]
    hidden = len(meta_cols) - max_cols
    new_rows = [r[:max_cols] for r in rows]
    return kept, new_rows, hidden

def _fmt_cell(value: Any, type_name: str, cell_limit: int) -> str:
    """
    Formats a cell value based on type and truncates according to `cell_limit`.
    """
    t = (type_name or "").upper()
    if value is None:
        return "NULL"
    try:
        if t in ("DECIMAL", "DOUBLE", "FLOAT"):
            return _truncate_text(f"{float(value):,.2f}", cell_limit)
        if t in ("INT", "BIGINT", "LONG"):
            return _truncate_text(f"{int(value):,}", cell_limit)
        return _truncate_text(str(value), cell_limit)
    except Exception:
        return _truncate_text(str(value), cell_limit)

def format_genie_answer_md(
    answer_json: Dict,
    *,
    rows_limit: int,
    cols_limit: int,
    cell_limit: int
) -> str:
    """
    Builds a markdown rendering for the Genie response, respecting user limits.
    """
    if "error" in answer_json:
        return f"⚠️ {answer_json['error']}"

    parts: List[str] = []

    if answer_json.get("query_description"):
        parts.append(f"**Query**: {answer_json['query_description']}")

    if "columns" in answer_json and "data" in answer_json:
        cols_meta = answer_json["columns"] or {}
        data = answer_json["data"] or {}

        raw_rows = data.get("data_array", []) or []
        rows, hidden_rows = _truncate_rows(raw_rows, rows_limit)

        # Enforce column limit
        meta_cols, rows, hidden_cols = _limit_cols(cols_meta, rows, cols_limit)

        if meta_cols:
            headers = [c.get("name", f"col{i+1}") for i, c in enumerate(meta_cols)]
            parts.append("| " + " | ".join(headers) + " |")
            parts.append("|" + "|".join(["---"] * len(headers)) + "|")

            for row in rows:
                formatted: List[str] = []
                for value, col in zip(row, meta_cols):
                    formatted.append(_fmt_cell(value, col.get("type_name") or "", cell_limit))
                parts.append("| " + " | ".join(formatted) + " |")

            notes: List[str] = []
            if hidden_rows:
                notes.append(f"{hidden_rows} hidden row(s)")
            if hidden_cols:
                notes.append(f"{hidden_cols} hidden column(s)")
            if notes:
                # Add guidance on how to increase limits via chat
                parts.append(
                    "\n_" + " • ".join(notes) + ". Refine your question to see fewer rows/columns._"
                    + "\n_" + "To see more, send: config cols=20 rows=200 (example)_"
                )
        else:
            parts.append("_No columns to display._")

    elif "message" in answer_json:
        parts.append(answer_json["message"] or "_No content._")

    else:
        parts.append("_No data available._")

    return "\n".join(parts)

def _is_genie_command(text: str) -> bool:
    """
    Detects legacy Genie prefixes (kept only for silent stripping).
    """
    t = text.strip().lower()
    return (
        t.startswith("/genie ")
        or t.startswith("/g ")
        or t.startswith("genie ")
        or t in ("/genie", "genie", "/g")
    )

def _strip_genie_prefix(text: str) -> str:
    """
    Strips optional '/genie', '/g', or 'genie' prefixes from the message.
    """
    t = re.sub(r"^/(genie|g)\s*:?\s*", "", text, flags=re.IGNORECASE).strip()
    t = re.sub(r"^genie\s*:?\s*", "", t, flags=re.IGNORECASE).strip()
    return t

def _help() -> str:
    """
    Returns the help text presented to end users.
    """
    return (
        f"**Databricks Genie Help** (v{VERSION})\n"
        "- Type your question directly (no prefix needed)\n"
        "- `config` → show your current limits\n"
        "- `config rows=100 cols=20 timeout=90` → adjust limits for your user\n"
        "- Fields: rows/linhas, cols/colunas/columns, chars, cell/cell_chars, timeout\n"
        f"- Default limits: rows={MAX_ROWS}, cols={GENIE_MAX_COLS}, chars/activity={GENIE_MAX_CHARS}, cell chars={GENIE_MAX_CELL_CHARS}\n"
        f"- Default timeout per call: {CALL_TIMEOUT_SECONDS}s\n"
    )

def _settings_summary(s: Dict[str, int]) -> str:
    """
    Short, human-friendly summary of current user limits.
    """
    return (f"Your current limits → rows={s['rows']}, cols={s['cols']}, "
            f"chars/activity={s['chars']}, cell chars={s['cell_chars']}, timeout={s['timeout']}s")

# ------------------------------------------------------------------------------
# Handlers (no echo; direct routing to Genie + per-user overrides via chat)
# ------------------------------------------------------------------------------

@AGENT_APP.conversation_update("membersAdded")
async def on_members_added(context: TurnContext, _state: TurnState):
    """
    Welcomes new members and explains how to use the bot,
    including how to adjust per-user limits via 'config'.
    """
    s = _default_settings()
    await context.send_activity(
        f"Hello! Genie is enabled. Version: {VERSION}\n"
        f"• Just **type your question** (no /genie needed).\n"
        f"• Example: `Top 5 customers by revenue`\n"
        f"• Commands: `config`, `config rows=100 cols=20 timeout=90`.\n"
        f"• {_settings_summary(s)}\n"
        f"• Tip: if you see _N hidden column(s)_, increase with `config cols=20`."
    )

@AGENT_APP.activity("message")
async def on_message(context: TurnContext, _state: TurnState):
    """
    Main message handler:
      - Shows version on 'version'
      - Supports 'config' to view/update per-user limits
      - Routes any other message directly to Genie (no echo)
    """
    text = (context.activity.text or "").strip()
    if not text:
        await context.send_activity("Send a message to get started. 🙂")
        return

    lower = text.lower()
    user_id = context.activity.from_property.id

    # Version info (does not block normal flow)
    if " version" in f" {lower} " or lower.strip() == "version":
        await context.send_activity(f"Running on version {VERSION}")

    # Configuration commands (view/update)
    if re.match(r'(?i)^\s*(config|settings?|set)(\b|:)', text) or _parse_config_overrides(text):
        overrides = _parse_config_overrides(text) or {}
        if overrides:
            s = _apply_overrides(user_id, overrides)
            await context.send_activity("✅ Settings updated.")
            await context.send_activity(_settings_summary(s))
        else:
            s = _get_settings_for(user_id)
            await context.send_activity(_settings_summary(s))
            await context.send_activity(
                "To adjust: `config rows=100 cols=20 timeout=90` • "
                "Fields: rows/linhas, cols/colunas/columns, chars, cell/cell_chars, timeout"
            )
        return

    # Help
    if lower in ("/genie help", "genie help", "/g help", "help", "/help"):
        await context.send_activity(_help())
        return

    # Genie configuration check
    if not DBX_ENABLED:
        await context.send_activity(
            "⚠️ Genie is not configured. Set `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, and `DATABRICKS_SPACE_ID` in the .env."
        )
        return

    # Backward-compat: silently strip old prefixes if users still type them
    question = _strip_genie_prefix(text)

    # Per-user settings
    s = _get_settings_for(user_id)

    # Track the user's conversation with Genie
    conv_id = _user_to_genie_conversation.get(user_id)

    # Call Genie with the user's timeout
    answer, new_conv = await ask_genie(
        question,
        DATABRICKS_SPACE_ID,
        conv_id,
        timeout=s["timeout"]
    )
    if new_conv:
        _user_to_genie_conversation[user_id] = new_conv

    # Render markdown respecting user limits (rows/cols/cell) and chunk by chars
    try:
        parsed = json.loads(answer)
    except Exception:
        parsed = {"message": answer}

    md = format_genie_answer_md(
        parsed,
        rows_limit=s["rows"],
        cols_limit=s["cols"],
        cell_limit=s["cell_chars"],
    )
    await _send_markdown(context, md, max_chars=s["chars"])
