"""MCP Integration: Bridges MCP servers into the LangClaw agent tool system.

MCPToolProvider reads server config, manages connections, discovers tools,
and wraps them as LangChain StructuredTool instances.
"""

import asyncio
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import create_model, Field

from .core.transport import StdioTransport, WebSocketTransport
from .core.connection import MCPConnection, ToolDefinition
from .core.registry import MCPRegistry
from .core.error import MCPError

import config as _cfg

logger = logging.getLogger(__name__)


class _AsyncBridge:
    """Runs an asyncio event loop in a dedicated background thread.

    Allows calling async code from synchronous contexts (e.g. inside
    ThreadPoolExecutor or when an event loop is already running).
    """

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro):
        """Run an async coroutine from sync code and return the result."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=60)

    def close(self):
        """Stop the event loop and join the thread."""
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


class MCPToolProvider:
    """Manages MCP server connections and provides LangChain-compatible tools.

    Usage:
        provider = MCPToolProvider()
        provider.initialize()        # connect to all configured servers
        tools = provider.build_tools()  # returns List[StructuredTool]
        ...
        provider.close_all()         # cleanup on shutdown
    """

    def __init__(self):
        self._bridge = _AsyncBridge()
        self._registry = MCPRegistry()
        self._initialized = False
        self._connected_servers: List[str] = []

    def initialize(self) -> None:
        """Read config, connect to MCP servers, discover tools."""
        servers_config = _cfg.get("mcp", "servers", default={})
        if not servers_config:
            logger.info("[MCP] No MCP servers configured")
            self._initialized = True
            return

        for server_name, server_cfg in servers_config.items():
            try:
                self._bridge.run(self._connect_server(server_name, server_cfg))
                self._connected_servers.append(server_name)
            except Exception as e:
                logger.warning(
                    "[MCP] Failed to connect to server '%s': %s. Skipping.",
                    server_name, e,
                )

        if self._connected_servers:
            self._bridge.run(self._registry.discover_all())
            all_tools = self._registry.list_all_tools()
            logger.info(
                "[MCP] Discovered %d tools from %d server(s): %s",
                len(all_tools),
                len(self._connected_servers),
                list(all_tools.keys()),
            )

        self._initialized = True

    async def _connect_server(self, name: str, cfg: dict) -> None:
        """Create transport, connect, initialize, and register a single server."""
        transport_type = cfg.get("transport", "stdio")

        if transport_type == "stdio":
            command = cfg.get("command")
            if not command:
                raise ValueError(f"Server '{name}': 'command' required for stdio")
            args = cfg.get("args", [])
            cwd = cfg.get("cwd")
            transport = StdioTransport(command=command, args=args, cwd=cwd)
        elif transport_type == "websocket":
            uri = cfg.get("uri")
            if not uri:
                raise ValueError(f"Server '{name}': 'uri' required for websocket")
            transport = WebSocketTransport(uri=uri)
        else:
            raise ValueError(f"Server '{name}': unknown transport '{transport_type}'")

        connection = MCPConnection(transport)
        await connection.connect()
        await connection.initialize(client_name="LangClaw", client_version="1.0")
        self._registry.register_connection(name, connection)
        logger.info("[MCP] Connected to server '%s' (%s)", name, transport_type)

    def build_tools(self) -> List[StructuredTool]:
        """Convert discovered MCP tools into LangChain StructuredTool instances.

        Tool names prefixed with 'mcp_' to avoid collisions.
        Each handler runs async MCP calls via _AsyncBridge.
        """
        if not self._initialized:
            return []

        tools = []
        for tool_name, metadata in self._registry.list_all_tools().items():
            prefixed_name = f"mcp_{tool_name}"
            description = (
                metadata.tool.description
                or f"MCP tool: {tool_name} (from {metadata.server_name})"
            )

            func = self._make_tool_handler(
                tool_name, metadata.server_name, self._bridge, self._registry
            )
            func.__name__ = prefixed_name
            func.__doc__ = description

            args_schema = _build_pydantic_schema(tool_name, metadata.tool.inputSchema)

            tools.append(StructuredTool(
                name=prefixed_name,
                description=description,
                func=func,
                args_schema=args_schema,
            ))

        return tools

    @staticmethod
    def _make_tool_handler(tool_name: str, server_name: str, bridge: _AsyncBridge,
                           registry: MCPRegistry):
        """Create a sync handler that bridges to async MCP call."""

        def handler(**kwargs) -> str:
            try:
                result = bridge.run(registry.call_tool(tool_name, kwargs))
                return _format_mcp_result(result)
            except MCPError as e:
                return f"MCP error calling '{tool_name}' on '{server_name}': {e}"
            except Exception as e:
                return f"Unexpected error calling MCP tool '{tool_name}': {e}"

        return handler

    def get_tool_summary(self) -> str:
        """Return a human-readable summary of available MCP tools for the system prompt."""
        if not self._initialized or not self._registry.list_all_tools():
            return ""

        lines = ["## MCP Tools\n", "You have access to external tools via MCP servers:\n"]
        for tool_name, metadata in self._registry.list_all_tools().items():
            desc = metadata.tool.description or "No description"
            lines.append(f"- **mcp_{tool_name}** ({metadata.server_name}): {desc}")

        return "\n".join(lines)

    def close_all(self) -> None:
        """Close all MCP connections gracefully."""
        try:
            self._bridge.run(self._registry.close_all())
        except Exception as e:
            logger.warning("[MCP] Error during close_all: %s", e)
        finally:
            self._connected_servers.clear()
            self._initialized = False
            self._bridge.close()


def _format_mcp_result(result: Any) -> str:
    """Format an MCP tool result into a string for the LLM."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        # MCP results often have a "content" field
        if "content" in result:
            content = result["content"]
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        parts.append(item["text"])
                    else:
                        parts.append(str(item))
                return "\n".join(parts)
            return str(content)
        return json.dumps(result, indent=2)
    return str(result)


def _build_pydantic_schema(tool_name: str, input_schema: dict):
    """Build a Pydantic BaseModel from an MCP tool's JSON Schema inputSchema.

    Returns None if the schema has no properties (no-arg tool).
    """
    properties = input_schema.get("properties", {})
    if not properties:
        return None

    required_fields = set(input_schema.get("required", []))

    field_definitions = {}
    for prop_name, prop_schema in properties.items():
        python_type = _json_type_to_python(prop_schema.get("type", "string"))
        description = prop_schema.get("description", "")
        is_required = prop_name in required_fields

        if is_required:
            field_definitions[prop_name] = (python_type, Field(description=description))
        else:
            default = prop_schema.get("default")
            field_definitions[prop_name] = (
                python_type, Field(default=default, description=description)
            )

    try:
        model_name = f"MCP_{tool_name.replace('-', '_').replace(' ', '_')}_Args"
        return create_model(model_name, **field_definitions)
    except Exception:
        return None


def _json_type_to_python(json_type: str) -> type:
    """Map JSON Schema type strings to Python types."""
    mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    return mapping.get(json_type, str)
