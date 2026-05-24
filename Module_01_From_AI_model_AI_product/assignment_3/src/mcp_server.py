import os
os.environ["FASTMCP_LOG_LEVEL"] = "ERROR"

import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from fastmcp import FastMCP
from src.data_loader import data_manager

mcp = FastMCP("Customer Service Data Analyst")

@mcp.tool()
def get_unique_values(column_name: str) -> list:
    """Retrieves all distinct values present within either the 'category' or 'intent' column."""
    df = data_manager.load_data()
    if column_name not in ['category', 'intent']:
        return [f"Error: Column must be 'category' or 'intent'. Got '{column_name}'"]
    return sorted(df[column_name].dropna().unique().tolist())


@mcp.tool()
def get_row_count(category: str = None, intent: str = None) -> dict:
    """Counts how many rows exist matching a given category or intent configuration."""
    df = data_manager.load_data()
    filtered_df = df
    if category:
        filtered_df = filtered_df[filtered_df['category'] == category]
    if intent:
        filtered_df = filtered_df[filtered_df['intent'] == intent]
    return {"matched_rows": len(filtered_df)}


@mcp.tool()
def get_sample_examples(category: str = None, intent: str = None, n_examples: int = 5) -> list:
    """Fetches a small set of real data rows for specific examples."""
    if not category and not intent:
        return [{
            "error": "You must specify a target 'category' or 'intent' value to pull examples. "
                     "If you do not know the valid options, please call 'get_unique_values' first."
        }]
    df = data_manager.load_data()
    filtered_df = df
    if category:
        filtered_df = filtered_df[filtered_df['category'] == category]
    if intent:
        filtered_df = filtered_df[filtered_df['intent'] == intent]
    if filtered_df.empty:
        return [{"message": f"No records found matching category='{category}' and intent='{intent}'."}]
    samples = filtered_df.head(n_examples)
    return samples[['instruction', 'category', 'intent', 'response']].to_dict(orient="records")


@mcp.tool()
def get_value_distribution(group_by_column: str, filter_column: str = None, filter_value: str = None) -> dict:
    """Calculates counts and distributions of entries across categories or intents."""
    df = data_manager.load_data()
    if group_by_column not in ['category', 'intent']:
        return {"error": "Group by column must be 'category' or 'intent'."}
    if filter_column and filter_value:
        df = df[df[filter_column] == filter_value]
    counts = df[group_by_column].value_counts().to_dict()
    return counts


@mcp.tool()
def filter_dataset(category: str = None, intent: str = None, limit: int = 1000) -> str:
    """Filters the entire dataset down by category and/or intent and returns structural status."""
    df = data_manager.load_data()
    filtered_df = df
    if category:
        filtered_df = filtered_df[filtered_df['category'] == category]
    if intent:
        filtered_df = filtered_df[filtered_df['intent'] == intent]
    total_count = len(filtered_df)
    truncated_df = filtered_df.head(limit)
    return f"Filter complete. Matches found: {total_count} rows. (Returned top {len(truncated_df)} rows for next step operations)."

 
if __name__ == "__main__":
    mcp.run(transport="stdio")