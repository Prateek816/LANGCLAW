# What exceptions do you need?
# - TransportError (base for transport problems)
# - ConnectionError (MCP-specific, not the built-in)
# - MCPError (when server returns error in JSON-RPC)

# What fields should MCPError have?
# - code (from JSON-RPC error.code)
# - message (from JSON-RPC error.message)
# - data (optional extra info)

"""MCP Error Classes"""


class MCPError(Exception):
    """Base exception for all MCP-related errors"""
    pass


class TransportError(MCPError):
    """Raised when transport fails (socket closed, subprocess crashed, etc.)"""
    
    def __init__(self, message: str, cause: Exception = None):
        super().__init__(message)
        self.cause = cause


class ConnectionError(MCPError):
    """Raised when MCP connection fails (handshake, state machine violation)"""
    
    def __init__(self, message: str, state: str = None):
        super().__init__(message)
        self.state = state


class ProtocolError(MCPError):
    """Raised when JSON-RPC protocol is violated (invalid response, missing fields)"""
    
    def __init__(self, message: str, response: dict = None):
        super().__init__(message)
        self.response = response


class ServerError(MCPError):
    """Raised when server returns JSON-RPC error response"""
    
    def __init__(self, code: int, message: str, data: dict = None):
        super().__init__(f"Server error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data or {}