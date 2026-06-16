"""MCP Connection: Single Server Session

MCPConnection wraps a Transport and adds MCP-aware logic:
- Request/response matching via futures
- State machine (created → connected → initialized → ready → closed)
- Background receiver task to dispatch messages
- Capability caching
"""

import asyncio
import json
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass
from .transport import BaseTransport
from .error import (
    TransportError,
    MCPConnectionError,
    ProtocolError,
    ServerError,
)


@dataclass
class ToolDefinition:
    """Represents a tool exposed by the server"""
    name: str
    description: str
    inputSchema: Dict[str, Any]


@dataclass
class ResourceDefinition:
    """Represents a resource exposed by the server"""
    uri: str
    name: str
    description: str
    mimeType: str


class MCPConnection:
    """Manages a single connection to an MCP server.
    
    Lifecycle:
        created → connected → initialized → ready → closed
    """
    
    def __init__(self, transport: BaseTransport):
        """Initialize connection with a transport.
        
        Args:
            transport: Transport instance (Stdio, WebSocket, etc.)
                      Must already be connected (or connect() called)
        """
        self.transport = transport
        
        # Request/response matching
        self.pending_requests: Dict[int, asyncio.Future] = {}
        self.request_id_counter = 0
        
        # Notification handlers: method name -> handler callable
        self.notification_handlers: Dict[str, Callable] = {}
        
        # Background receiver task
        self.bg_task: Optional[asyncio.Task] = None
        self.bg_task_lock = asyncio.Lock()
        
        # State machine
        self.state = "created"  # created, connected, initialized, ready, closed
        
        # Server info from initialize response
        self.server_info: Optional[Dict] = None
        
        # Cached capabilities
        self.tools: Dict[str, ToolDefinition] = {}
        self.resources: Dict[str, ResourceDefinition] = {}
        self.prompts: List[str] = []
    
    # ========== State Machine & Lifecycle ==========
    
    async def connect(self) -> None:
        """Establish transport connection (for transports that need explicit connect)."""
        if self.state != "created":
            raise MCPConnectionError(f"Cannot connect from state {self.state}", state=self.state)
        
        try:
            # Some transports (like StdioTransport) need explicit connect()
            if hasattr(self.transport, 'connect'):
                await self.transport.connect()
            self.state = "connected"
        except TransportError as e:
            self.state = "closed"
            raise MCPConnectionError(f"Failed to connect transport: {e}", state="created")
    
    async def initialize(self, client_name: str = "LangClaw", client_version: str = "1.0") -> Dict:
        """Perform MCP initialization handshake.
        
        Sequence:
        1. Send initialize request
        2. Receive initialize response
        3. Validate protocol version
        4. Start background receiver task
        5. Send initialized notification
        6. Mark as ready
        
        Args:
            client_name: Name of client (appears in server logs)
            client_version: Client version string
        
        Returns:
            Server info dict
        
        Raises:
            ConnectionError: If already initialized or transport not ready
            ProtocolError: If server response is invalid
        """
        if self.state not in ["connected"]:
            raise MCPConnectionError(
                f"Cannot initialize from state {self.state}. Call connect() first.",
                state=self.state
            )
        
        # 1. Send initialize request
        try:
            response = await self._send_request(
                method="initialize",
                params={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": client_name,
                        "version": client_version,
                    }
                }
            )
        except Exception as e:
            self.state = "closed"
            raise MCPConnectionError(f"Initialize request failed: {e}", state="connected")
        
        # 2. Validate response
        if "protocolVersion" not in response:
            raise ProtocolError("Server initialize response missing protocolVersion", response)
        
        # 3. Store server info
        self.server_info = response
        
        # 4. Start background receiver BEFORE sending initialized
        # This ensures we don't miss any notifications
        try:
            async with self.bg_task_lock:
                if self.bg_task is None:
                    self.bg_task = asyncio.create_task(self._background_receiver())
        except Exception as e:
            self.state = "closed"
            raise MCPConnectionError(f"Failed to start background receiver: {e}", state="connected")
        
        # 5. Send initialized notification (one-way, no response)
        try:
            await self._send_notification("initialized", params={})
            self.state = "ready"
        except Exception as e:
            self.state = "closed"
            raise MCPConnectionError(f"Failed to send initialized: {e}", state="connected")
        
        return response
    
    async def close(self) -> None:
        """Gracefully close connection and cleanup."""
        if self.state == "closed":
            return
        
        self.state = "closed"
        
        # Cancel background task
        if self.bg_task and not self.bg_task.done():
            self.bg_task.cancel()
            try:
                await self.bg_task
            except asyncio.CancelledError:
                pass
        
        # Close transport
        try:
            await self.transport.close()
        except Exception:
            pass
        
        # Reject pending requests
        for future in self.pending_requests.values():
            if not future.done():
                future.set_exception(MCPConnectionError("Connection closed", state="closed"))
        self.pending_requests.clear()
    
    # ========== Request/Response Handling ==========
    
    def _get_next_request_id(self) -> int:
        """Generate next unique request ID."""
        self.request_id_counter += 1
        return self.request_id_counter
    
    async def _send_request(self, method: str, params: Dict = None) -> Any:
        """Send a JSON-RPC request and wait for response.
        
        Args:
            method: JSON-RPC method name
            params: Method parameters
        
        Returns:
            The result field from response
        
        Raises:
            ServerError: If server returns error
            TransportError: If send fails
            ConnectionError: If state is not ready
        """
        if self.state != "ready":
            raise MCPConnectionError(f"Cannot send request from state {self.state}", state=self.state)
        
        # Get unique ID for this request
        request_id = self._get_next_request_id()
        
        # Create a future to wait for the response
        future: asyncio.Future = asyncio.Future()
        self.pending_requests[request_id] = future
        
        # Build JSON-RPC request
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params:
            request["params"] = params
        
        try:
            # Send via transport
            await self.transport.send(request)
            
            # Wait for response (future will be completed by bg_receiver)
            result = await future
            return result
        except asyncio.CancelledError:
            # Connection being closed
            raise MCPConnectionError("Request cancelled (connection closing)", state=self.state)
        except Exception as e:
            raise
        finally:
            # Cleanup
            self.pending_requests.pop(request_id, None)
    
    async def _send_notification(self, method: str, params: Dict = None) -> None:
        """Send a JSON-RPC notification (no response expected).
        
        Args:
            method: JSON-RPC method name
            params: Method parameters
        
        Raises:
            TransportError: If send fails
        """
        notification = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params:
            notification["params"] = params
        
        await self.transport.send(notification)
    
    # ========== Background Receiver Task ==========
    
    async def _background_receiver(self) -> None:
        """Background task that continuously receives messages.
        
        - Receives messages from transport
        - Matches responses to pending requests via ID
        - Routes notifications to handlers
        - Runs until connection closes
        """
        try:
            while self.state != "closed":
                try:
                    # Block until message arrives
                    message = await self.transport.recv()
                    
                    # Dispatch based on message type
                    if "id" in message:
                        # Response to a request
                        await self._handle_response(message)
                    else:
                        # Notification (no id)
                        await self._handle_notification(message)
                
                except TransportError as e:
                    # Transport failed - close connection
                    print(f"[MCP] Transport error in background receiver: {e}")
                    self.state = "closed"
                    break
                except Exception as e:
                    print(f"[MCP] Unexpected error in background receiver: {e}")
                    continue
        
        except asyncio.CancelledError:
            # Connection being closed - normal exit
            pass
        finally:
            self.state = "closed"
    
    async def _handle_response(self, message: Dict) -> None:
        """Handle a JSON-RPC response message.
        
        Looks up the corresponding future and completes it with:
        - result value if "result" present
        - ServerError if "error" present
        """
        request_id = message.get("id")
        
        # Find the waiting future
        future = self.pending_requests.get(request_id)
        if future is None:
            print(f"[MCP] Received response for unknown request ID {request_id}")
            return
        
        # Check if response contains error
        if "error" in message:
            error_obj = message["error"]
            exc = ServerError(
                code=error_obj.get("code", -1),
                message=error_obj.get("message", "Unknown error"),
                data=error_obj.get("data", {})
            )
            future.set_exception(exc)
        elif "result" in message:
            # Success - set result
            future.set_result(message["result"])
        else:
            # Malformed response
            future.set_exception(
                ProtocolError("Response has neither 'result' nor 'error'", message)
            )
    
    async def _handle_notification(self, message: Dict) -> None:
        """Handle a JSON-RPC notification (unsolicited message from server).
        
        Routes to registered handler if one exists.
        """
        method = message.get("method")
        params = message.get("params", {})
        
        # Look for registered handler
        handler = self.notification_handlers.get(method)
        if handler:
            try:
                result = handler(params)
                # If handler is async, await it
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                print(f"[MCP] Error in notification handler for {method}: {e}")
        else:
            print(f"[MCP] No handler for notification: {method}")
    
    # ========== Tool & Resource Methods ==========
    
    async def list_tools(self) -> List[ToolDefinition]:
        """Request list of available tools from server.
        
        Returns:
            List of ToolDefinition objects
        
        Raises:
            ServerError: If server returns error
        """
        result = await self._send_request("tools/list", params={})
        
        # Parse response and cache tools
        self.tools.clear()
        tools = []
        for tool_dict in result.get("tools", []):
            tool = ToolDefinition(
                name=tool_dict["name"],
                description=tool_dict.get("description", ""),
                inputSchema=tool_dict.get("inputSchema", {})
            )
            self.tools[tool.name] = tool
            tools.append(tool)
        
        return tools
    
    async def call_tool(self, name: str, arguments: Dict = None) -> Any:
        """Call a tool on the server.
        
        Args:
            name: Tool name
            arguments: Tool input arguments
        
        Returns:
            Tool result
        
        Raises:
            ServerError: If tool execution fails
        """
        result = await self._send_request(
            "tools/call",
            params={
                "name": name,
                "arguments": arguments or {}
            }
        )
        return result
    
    async def list_resources(self) -> List[ResourceDefinition]:
        """Request list of available resources from server.
        
        Returns:
            List of ResourceDefinition objects
        """
        result = await self._send_request("resources/list", params={})
        
        # Parse response and cache resources
        self.resources.clear()
        resources = []
        for res_dict in result.get("resources", []):
            res = ResourceDefinition(
                uri=res_dict["uri"],
                name=res_dict.get("name", ""),
                description=res_dict.get("description", ""),
                mimeType=res_dict.get("mimeType", "text/plain")
            )
            self.resources[res.uri] = res
            resources.append(res)
        
        return resources
    
    async def read_resource(self, uri: str) -> str:
        """Read the contents of a resource.
        
        Args:
            uri: Resource URI
        
        Returns:
            Resource content as string
        """
        result = await self._send_request(
            "resources/read",
            params={"uri": uri}
        )
        return result.get("contents", "")
    
    # ========== Notification Handler Registration ==========
    
    def register_notification_handler(self, method: str, handler: Callable) -> None:
        """Register a handler for incoming notifications.
        
        Args:
            method: Notification method name (e.g., "resource_updated")
            handler: Callable that takes params dict. Can be async.
        
        Example:
            def on_resource_update(params):
                print(f"Resource updated: {params}")
            
            conn.register_notification_handler("resource_updated", on_resource_update)
        """
        self.notification_handlers[method] = handler
    
    def unregister_notification_handler(self, method: str) -> None:
        """Unregister a notification handler."""
        self.notification_handlers.pop(method, None)