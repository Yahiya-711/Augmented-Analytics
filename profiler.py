import pandas as pd

def profile_dataframe(df: pd.DataFrame) -> dict:
    """
    Performs a full statistical analysis of a DataFrame.
    """
    # Basic Statistics
    numerical_cols = df.select_dtypes(include=["number"]).columns
    stats = df[numerical_cols].describe().to_dict()

    # Outlier Detection
    outliers_summary = {}
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_condition = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
        outliers_summary[col] = int(outlier_condition.sum())

    # Categorical Analysis
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    categorical_analysis = {}
    for col in categorical_cols:
        categorical_analysis[col] = {
            "value_counts": df[col].value_counts().to_dict(),
            "unique_values_count": int(df[col].nunique())
        }
        
    # Combine into a single dictionary
    full_profile = {
        "basic_statistics": stats,
        "outliers_count": outliers_summary,
        "categorical_analysis": categorical_analysis
    }
    print("✅ DataFrame profiled successfully.")
    return full_profile