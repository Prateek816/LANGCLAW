"""MCP Registry: Multi-Server Management

MCPRegistry manages multiple MCPConnection instances and provides:
- Tool discovery across all connected servers
- Tool routing (which server has tool X?)
- Unified tool calling interface
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from .connection import MCPConnection, ToolDefinition
from .error import MCPConnectionError, ServerError


@dataclass
class ToolMetadata:
    """Tool with metadata about which server provides it"""
    tool: ToolDefinition
    server_name: str
    connection: MCPConnection


class MCPRegistry:
    """Registry of all connected MCP servers and their capabilities."""
    
    def __init__(self):
        """Initialize empty registry."""
        self.connections: Dict[str, MCPConnection] = {}
        self.tools_index: Dict[str, ToolMetadata] = {}
    
    # ========== Connection Management ==========
    
    def register_connection(self, name: str, connection: MCPConnection) -> None:
        """Register a new MCP server connection.
        
        Args:
            name: Unique identifier for this server (e.g., "github-mcp", "web-search")
            connection: MCPConnection instance (should be in "ready" state)
        
        Raises:
            ConnectionError: If connection is not ready
        """
        if connection.state != "ready":
            raise MCPConnectionError(
                f"Cannot register connection in state {connection.state}. "
                "Connection must be ready (call initialize() first).",
                state=connection.state
            )
        
        self.connections[name] = connection
    
    def unregister_connection(self, name: str) -> None:
        """Unregister a server connection.
        
        Removes all tools provided by this server from the index.
        
        Args:
            name: Server identifier
        """
        if name not in self.connections:
            return
        
        connection = self.connections.pop(name)
        
        # Remove all tools from this server from the index
        to_remove = [
            tool_name for tool_name, metadata in self.tools_index.items()
            if metadata.server_name == name
        ]
        for tool_name in to_remove:
            self.tools_index.pop(tool_name)
    
    async def close_all(self) -> None:
        """Close all connections gracefully."""
        for connection in self.connections.values():
            try:
                await connection.close()
            except Exception as e:
                print(f"[MCP] Error closing connection: {e}")
        
        self.connections.clear()
        self.tools_index.clear()
    
    # ========== Tool Discovery & Indexing ==========
    
    async def discover_tools(self, server_name: str) -> List[ToolDefinition]:
        """Discover tools from a specific server and add to index.
        
        Args:
            server_name: Server identifier
        
        Returns:
            List of ToolDefinition objects
        
        Raises:
            ConnectionError: If server not registered
            ServerError: If server returns error
        """
        if server_name not in self.connections:
            raise MCPConnectionError(f"Server {server_name} not registered")
        
        connection = self.connections[server_name]
        
        # Request tool list
        tools = await connection.list_tools()
        
        # Index them (map tool name -> metadata)
        for tool in tools:
            self.tools_index[tool.name] = ToolMetadata(
                tool=tool,
                server_name=server_name,
                connection=connection
            )
        
        return tools
    
    async def discover_all(self) -> Dict[str, List[ToolDefinition]]:
        """Discover tools from all registered servers.
        
        Returns:
            Dict mapping server name -> list of tools
        """
        result = {}
        for server_name in self.connections.keys():
            try:
                tools = await self.discover_tools(server_name)
                result[server_name] = tools
            except Exception as e:
                print(f"[MCP] Error discovering tools from {server_name}: {e}")
                result[server_name] = []
        
        return result
    
    # ========== Tool Lookup & Calling ==========
    
    def get_tool(self, name: str) -> Optional[ToolMetadata]:
        """Look up a tool by name.
        
        Args:
            name: Tool name
        
        Returns:
            ToolMetadata if found, None otherwise
        """
        return self.tools_index.get(name)
    
    def list_all_tools(self) -> Dict[str, ToolMetadata]:
        """Get all indexed tools.
        
        Returns:
            Dict mapping tool name -> ToolMetadata
        """
        return dict(self.tools_index)
    
    async def call_tool(self, name: str, arguments: Dict = None) -> Any:
        """Call a tool by name (router finds which server has it).
        
        Args:
            name: Tool name
            arguments: Tool input arguments
        
        Returns:
            Tool result
        
        Raises:
            ConnectionError: If tool not found
            ServerError: If tool execution fails
        """
        metadata = self.get_tool(name)
        if not metadata:
            raise MCPConnectionError(f"Tool '{name}' not found in any server")
        
        # Call via the server connection
        return await metadata.connection.call_tool(name, arguments)
    
    # ========== Convenience Methods ==========
    
    def get_connection(self, name: str) -> Optional[MCPConnection]:
        """Get a connection by server name.
        
        Args:
            name: Server identifier
        
        Returns:
            MCPConnection if registered, None otherwise
        """
        return self.connections.get(name)
    
    def list_servers(self) -> List[str]:
        """Get names of all registered servers.
        
        Returns:
            List of server names
        """
        return list(self.connections.keys())
    
    def get_server_tools(self, server_name: str) -> List[str]:
        """Get tool names provided by a specific server.
        
        Args:
            server_name: Server identifier
        
        Returns:
            List of tool names from this server
        """
        return [
            tool_name for tool_name, metadata in self.tools_index.items()
            if metadata.server_name == server_name
        ]