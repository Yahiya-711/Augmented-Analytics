import pandas as pd

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans a DataFrame by imputing missing values based on column type.
    - Fills numerical columns with the median.
    - Fills categorical and datetime columns with the mode.
    """
    cleaned_df = df.copy()
    for column in cleaned_df.columns:
        if cleaned_df[column].isnull().any():
            if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                median_val = cleaned_df[column].median()
                cleaned_df[column].fillna(median_val, inplace=True)
            else: # For object, category, and datetime types
                mode_val = cleaned_df[column].mode()[0]
                cleaned_df[column].fillna(mode_val, inplace=True)
    print("✅ DataFrame cleaned successfully.")
    return cleaned_df