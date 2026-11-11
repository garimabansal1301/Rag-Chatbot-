"""Model Context Protocol (MCP): a local stdio server for the calculator and current_datetime tools, and the
client that connects the chat engine to it. Off unless MCP_ENABLED=true; see README.md ("MCP").

  app/mcp/server.py  the FastMCP server (run with: uv run python -m app.mcp.server)
  app/mcp/client.py  MCPToolset: one long-lived stdio session, its tools wrapped for the synchronous graph
"""
