import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastmcp import Client
from mcp_server import mcp

async def main():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print("Available tools:", [t.name for t in tools])

        result = await client.call_tool("get_unique_values", {"column_name": "category"})
        print("\nUnique categories:", result.data)

        result = await client.call_tool("get_row_count", {"category": "REFUND"})
        print("\nREFUND row count:", result.data)

        result = await client.call_tool("get_sample_examples", {
            "intent": "get_refund",
            "n_examples": 2
        })
        print("\nSample get_refund examples:", result.data)

if __name__ == "__main__":
    asyncio.run(main())