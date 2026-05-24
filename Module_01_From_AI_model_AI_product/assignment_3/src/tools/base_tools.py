import pandas as pd
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from src.data_loader import data_manager

# --- Pydantic Argument Schemas ---

class GetUniqueValuesInput(BaseModel):
    column_name: str = Field(
        ..., 
        description="The target column to check. Must be exactly 'category' or 'intent'."
    )

class FilterDatasetInput(BaseModel):
    category: Optional[str] = Field(
        None, 
        description="Optional high-level category string to filter by (e.g., 'ORDER', 'ACCOUNT'). Case-sensitive."
    )
    intent: Optional[str] = Field(
        None, 
        description="Optional exact intent string to filter by (e.g., 'cancel_order', 'get_refund')."
    )
    limit: int = Field(
        1000, 
        description="Max rows to extract to prevent token overflow. Default is 1000."
    )

class SampleExamplesInput(BaseModel):
    category: Optional[str] = Field(None, description="Filter by exact category value.")
    intent: Optional[str] = Field(None, description="Filter by exact intent value.")
    n_examples: int = Field(5, description="The specific number of sample rows to retrieve.")

class GetDistributionInput(BaseModel):
    group_by_column: str = Field(
        ..., 
        description="The primary column to group by. Must be 'category' or 'intent'."
    )
    filter_column: Optional[str] = Field(
        None, 
        description="Optional column to filter on before taking distributions (e.g., 'category')."
    )
    filter_value: Optional[str] = Field(
        None, 
        description="The exact value to filter the filter_column by (e.g., 'ACCOUNT')."
    )

# --- Tool Implementations ---

@tool("get_unique_values", args_schema=GetUniqueValuesInput)
def get_unique_values(column_name: str) -> List[str]:
    """Retrieves all distinct values present within either the 'category' or 'intent' column."""
    df = data_manager.load_data()
    if column_name not in ['category', 'intent']:
        return [f"Error: Column must be 'category' or 'intent'. Got '{column_name}'"]
    return sorted(df[column_name].dropna().unique().tolist())


@tool("filter_dataset", args_schema=FilterDatasetInput)
def filter_dataset(category: Optional[str] = None, intent: Optional[str] = None, limit: int = 1000) -> str:
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


@tool("get_row_count", args_schema=FilterDatasetInput)
def get_row_count(category: Optional[str] = None, intent: Optional[str] = None, limit: int = 1000) -> Dict[str, Any]:
    """Counts how many rows exist matching a given category or intent configuration."""
    df = data_manager.load_data()
    filtered_df = df
    if category:
        filtered_df = filtered_df[filtered_df['category'] == category]
    if intent:
        filtered_df = filtered_df[filtered_df['intent'] == intent]
    return {"matched_rows": len(filtered_df)}


@tool("get_sample_examples", args_schema=SampleExamplesInput)
def get_sample_examples(category: Optional[str] = None, intent: Optional[str] = None, n_examples: int = 5) -> Any:
    """Fetches a small set of real data rows (containing instructions and responses) for specific examples."""
    # Ensure n_examples is handled strictly as an integer
    try:
        n_examples = int(n_examples)
    except ValueError:
        n_examples = 5

    # Safeguard: Force the agent to pick a lane. If it passes nothing, prevent raw dumping.
    if not category and not intent:
        return (
            "Error: You must specify a target 'category' or 'intent' value to pull examples. "
            "If you do not know the valid options, please call 'get_unique_values' first."
        )

    df = data_manager.load_data()
    filtered_df = df
    
    if category:
        filtered_df = filtered_df[filtered_df['category'] == category]
    if intent:
        filtered_df = filtered_df[filtered_df['intent'] == intent]
        
    if filtered_df.empty:
        return f"No records found matching category='{category}' and intent='{intent}'."
        
    samples = filtered_df.head(n_examples)
    return samples[['instruction', 'category', 'intent', 'response']].to_dict(orient="records")


@tool("get_value_distribution", args_schema=GetDistributionInput)
def get_value_distribution(group_by_column: str, filter_column: Optional[str] = None, filter_value: Optional[str] = None) -> Dict[str, int]:
    """Calculates counts and distributions of entries across categories or intents."""
    df = data_manager.load_data()
    if group_by_column not in ['category', 'intent']:
        return {"error": "Group by column must be 'category' or 'intent'."}
        
    if filter_column and filter_value:
        df = df[df[filter_column] == filter_value]
        
    counts = df[group_by_column].value_counts().to_dict()
    return counts

# Export a registry list of our tools for the graph engine to bind
ALL_TOOLS = [get_unique_values, filter_dataset, get_row_count, get_sample_examples, get_value_distribution]