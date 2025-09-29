import pandas as pd
from langchain.tools import StructuredTool

class InferenceToolSet:
    """A class that holds a DataFrame and exposes statistical methods as tools."""
    
    def __init__(self, df: pd.DataFrame):
        """Initializes the toolset with a specific DataFrame."""
        self.df = df.copy()
        print("✅ InferenceToolSet initialized with a DataFrame.")

    def get_basic_statistics(self) -> dict:
        """
        Calculates basic descriptive statistics (mean, median, std, min, max) 
        for all numerical columns in the DataFrame.
        """
        numerical_cols = self.df.select_dtypes(include=["number"]).columns
        if len(numerical_cols) == 0:
            return {"message": "No numerical columns found."}
        
        stats = self.df[numerical_cols].describe().to_dict()
        return stats

    def detect_outliers(self) -> dict:
        """
        Detects outliers in numerical columns using the IQR (Interquartile Range) method.
        Returns a dictionary with the count of outliers for each numerical column.
        """
        numerical_cols = self.df.select_dtypes(include=["number"]).columns
        if len(numerical_cols) == 0:
            return {"message": "No numerical columns found for outlier detection."}
        
        outliers_summary = {}
        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_condition = (self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))
            outliers_summary[col] = int(outlier_condition.sum())
            
        return {"outliers_count": outliers_summary}

    def analyze_categorical_data(self) -> dict:
        """
        Analyzes categorical columns by calculating value counts and the number of unique values.
        """
        categorical_cols = self.df.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) == 0:
            return {"message": "No categorical columns found."}
        
        categorical_analysis = {}
        for col in categorical_cols:
            categorical_analysis[col] = {
                "value_counts": self.df[col].value_counts().to_dict(),
                "unique_values_count": int(self.df[col].nunique())
            }
        return categorical_analysis
            
    def get_tools(self):
        """Returns a list of Tool objects for the agent."""
        
        stats_tool = StructuredTool.from_function(
            func=self.get_basic_statistics,
            name="get_basic_statistics",
            description="Calculates and returns basic descriptive statistics for numerical columns."
        )
        
        outlier_tool = StructuredTool.from_function(
            func=self.detect_outliers,
            name="detect_outliers",
            description="Detects and returns the count of outliers in numerical columns using the IQR method."
        )
        
        categorical_tool = StructuredTool.from_function(
            func=self.analyze_categorical_data,
            name="analyze_categorical_data",
            description="Analyzes categorical columns, returning value counts and unique value counts."
        )
        
        return [stats_tool, outlier_tool, categorical_tool]
