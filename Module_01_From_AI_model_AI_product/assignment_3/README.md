## MCP Server

The project exposes its analytical tools via an MCP server built with FastMCP.

### Start the Server

```bash
pip install fastmcp
python src/mcp_server.py
```

### Connect a Client

```python
from fastmcp import Client
from mcp_server import mcp
import asyncio

async def main():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print("Available:", [t.name for t in tools])

        result = await client.call_tool("get_unique_values", {"column_name": "category"})
        print("Categories:", result.data)

asyncio.run(main())
```

### Available MCP Tools

| Tool | Description |
|---|---|
| `get_unique_values` | Distinct values in 'category' or 'intent' |
| `get_row_count` | Count rows matching filters |
| `get_sample_examples` | Pull real data examples |
| `get_value_distribution` | Distribution counts |
| `filter_dataset` | Filter and return match summary |