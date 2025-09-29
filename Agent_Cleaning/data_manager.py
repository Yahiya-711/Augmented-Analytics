import pandas as pd

class DataFrameManager:
    """A simple class to hold and manage the state of the DataFrame."""
    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()
        print("✅ DataFrameManager initialized.")

    def get_df(self) -> pd.DataFrame:
        """Returns a copy of the current DataFrame."""
        return self._df.copy()

    def update_df(self, new_df: pd.DataFrame):
        """Updates the internal DataFrame."""
        self._df = new_df.copy()
        print("✅ DataFrame has been updated by a tool.")

# This global instance will be shared across the agent's modules.
# It gets initialized by the script that runs the agent.
df_manager: DataFrameManager = None
