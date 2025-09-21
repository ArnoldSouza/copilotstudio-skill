# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# enable logging for Microsoft Agents library
# for more information, see README.md for Quickstart Agent
import logging
ms_agents_logger = logging.getLogger("microsoft_agents")
ms_agents_logger.addHandler(logging.StreamHandler())
ms_agents_logger.setLevel(logging.INFO)

from os import environ

from microsoft_agents.hosting.core import AgentApplication
from microsoft_agents.hosting.aiohttp import (
    start_agent_process,
    jwt_authorization_middleware,
    CloudAdapter,
)
from aiohttp.web import Request, Response, Application, run_app, static
from aiohttp import web

from .agent import AGENT_APP, CONNECTION_MANAGER


async def init_func():
    app = web.Application()
    # aqui você adiciona rotas
    app.router.add_get("/", lambda request: web.Response(text="Hello World"))
    return app

async def healthz(req: Request) -> Response:
    return Response(text="ok")


async def entry_point(req: Request) -> Response:
    agent: AgentApplication = req.app["agent_app"]
    adapter: CloudAdapter = req.app["adapter"]
    return await start_agent_process(
        req,
        agent,
        adapter,
    )


APP_WRAPPER = Application()
APP_WRAPPER.add_routes([static("/public", "./public")])
APP_WRAPPER.router.add_get("/healthz", healthz)

APP = Application(middlewares=[jwt_authorization_middleware])
APP.router.add_post("/messages", entry_point)

APP_WRAPPER.add_subapp("/api", APP)

APP["agent_configuration"] = CONNECTION_MANAGER.get_default_connection_configuration()
APP["agent_app"] = AGENT_APP
APP["adapter"] = AGENT_APP.adapter

try:
    port = int(environ.get("PORT", 3978))
    run_app(APP_WRAPPER, host="0.0.0.0", port=port)
except Exception as error:
    raise error
