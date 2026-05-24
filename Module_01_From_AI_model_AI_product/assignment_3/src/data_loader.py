import os
import pandas as pd
from typing import Optional

class DatasetLoader:
    def __init__(self, file_path: Optional[str] = None):
        """Initializes the loader with a default or custom file path."""
        self.file_path = file_path or os.path.join(
            os.path.dirname(__file__), "..", "data", "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
        )
        self.df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Loads the Bitext CSV dataset into a Pandas DataFrame."""
        if self.df is not None:
            return self.df
            
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"Dataset not found at {self.file_path}."
            )
            
        self.df = pd.read_csv(self.file_path)
        return self.df

    def get_schema_summary(self) -> dict:
        """Returns basic column and structural metadata."""
        df = self.load_data()
        return {
            "columns": list(df.columns),
            "shape": df.shape,
            "sample": df.head(2).to_dict(orient="records")
        }

# Instantiate global data manager instance to resolve import errors
data_manager = DatasetLoader()