"""MCP Transport Abstraction Layer

Transports handle the low-level communication (stdio, websocket, http).
They read/write raw JSON text and handle framing.
"""

import json
import asyncio
from abc import ABC, abstractmethod
from typing import Optional
from .error import TransportError


class BaseTransport(ABC):
    """Abstract base class for all MCP transports.
    
    A transport is responsible for:
    - Sending JSON-RPC messages to the server
    - Receiving JSON-RPC messages from the server
    - Handling encoding/decoding and framing
    """
    
    @abstractmethod
    async def send(self, message: dict) -> None:
        """Send a JSON-RPC message to the server.
        
        Args:
            message: Dict with jsonrpc, method/result, id (optional), params, etc.
        
        Raises:
            TransportError: If send fails
        """
        pass
    
    @abstractmethod
    async def recv(self) -> dict:
        """Receive and parse a JSON-RPC message from server.
        
        Returns:
            Dict parsed from JSON
        
        Raises:
            TransportError: If receive fails or connection closes
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the transport gracefully."""
        pass
    
    @abstractmethod
    def is_open(self) -> bool:
        """Check if transport is still connected."""
        pass


class StdioTransport(BaseTransport):
    """Stdio transport for MCP servers running as subprocesses.
    
    The server is spawned as a subprocess. We communicate via:
    - stdin: we write JSON-RPC messages here
    - stdout: we read JSON-RPC messages from here
    """
    
    def __init__(self, command: str, args: list = None, cwd: str = None):
        """Initialize stdio transport for a subprocess.
        
        Args:
            command: Path to executable (e.g., "python", "/usr/bin/node")
            args: List of arguments to pass to executable
            cwd: Working directory for subprocess
        """
        self.command = command
        self.args = args or []
        self.cwd = cwd
        self.process: Optional[asyncio.subprocess.Process] = None
        self._open = False
    
    async def connect(self) -> None:
        """Spawn the subprocess and establish pipes."""
        try:
            self.process = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
            )
            self._open = True
        except Exception as e:
            raise TransportError(f"Failed to spawn subprocess: {e}", cause=e)
    
    async def send(self, message: dict) -> None:
        """Send JSON-RPC message via subprocess stdin."""
        if not self._open or not self.process:
            raise TransportError("Transport not open")
        
        try:
            # Encode message as JSON, add newline for framing
            line = json.dumps(message, separators=(',', ':')) + '\n'
            self.process.stdin.write(line.encode('utf-8'))
            await self.process.stdin.drain()
        except Exception as e:
            self._open = False
            raise TransportError(f"Failed to send message: {e}", cause=e)
    
    async def recv(self) -> dict:
        """Receive and parse JSON-RPC message from subprocess stdout."""
        if not self._open or not self.process:
            raise TransportError("Transport not open")
        
        try:
            # Read line from stdout (blocks until newline)
            line = await self.process.stdout.readline()
            
            # If EOF reached, connection is closed
            if not line:
                self._open = False
                raise TransportError("Server closed connection (EOF)")
            
            # Decode and parse JSON
            message = json.loads(line.decode('utf-8'))
            return message
        except json.JSONDecodeError as e:
            raise TransportError(f"Invalid JSON from server: {e}", cause=e)
        except Exception as e:
            self._open = False
            raise TransportError(f"Failed to receive message: {e}", cause=e)
    
    async def close(self) -> None:
        """Close pipes and terminate subprocess."""
        if self.process:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                # Wait for process to exit (with timeout)
                await asyncio.wait_for(self.process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                # Force kill if it doesn't exit gracefully
                self.process.kill()
                await self.process.wait()
            except Exception:
                pass
        
        self._open = False
    
    def is_open(self) -> bool:
        """Check if subprocess is still running."""
        if not self.process or not self._open:
            return False
        return self.process.returncode is None


class WebSocketTransport(BaseTransport):
    """WebSocket transport for MCP servers running on the network.
    
    Connects to a WebSocket endpoint and exchanges JSON-RPC messages.
    """
    
    def __init__(self, uri: str):
        """Initialize WebSocket transport.
        
        Args:
            uri: WebSocket URI (e.g., "ws://localhost:8765")
        """
        self.uri = uri
        self.ws = None
        self._open = False
    
    async def connect(self) -> None:
        """Connect to WebSocket server."""
        try:
            import websockets
            self.ws = await websockets.asyncio.client.connect(self.uri)
            self._open = True
        except ImportError:
            raise TransportError("websockets library not installed")
        except Exception as e:
            raise TransportError(f"Failed to connect to {self.uri}: {e}", cause=e)
    
    async def send(self, message: dict) -> None:
        """Send JSON-RPC message via WebSocket."""
        if not self._open or not self.ws:
            raise TransportError("Transport not open")
        
        try:
            line = json.dumps(message, separators=(',', ':'))
            await self.ws.send(line)
        except Exception as e:
            self._open = False
            raise TransportError(f"Failed to send message: {e}", cause=e)
    
    async def recv(self) -> dict:
        """Receive and parse JSON-RPC message from WebSocket."""
        if not self._open or not self.ws:
            raise TransportError("Transport not open")
        
        try:
            line = await self.ws.recv()
            message = json.loads(line)
            return message
        except json.JSONDecodeError as e:
            raise TransportError(f"Invalid JSON from server: {e}", cause=e)
        except Exception as e:
            self._open = False
            raise TransportError(f"Failed to receive message: {e}", cause=e)
    
    async def close(self) -> None:
        """Close WebSocket connection."""
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
        self._open = False
    
    def is_open(self) -> bool:
        """Check if WebSocket is still connected."""
        return self._open and self.ws is not None