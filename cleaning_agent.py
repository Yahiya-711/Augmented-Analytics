import pandas as pd
from langchain.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from .data_manager import DataFrameManager

class CleaningToolSet:
    """A class that holds the dataframe manager and exposes cleaning methods as tools."""
    
    def __init__(self, df_manager: DataFrameManager):
        """Initializes the toolset with a specific DataFrameManager instance."""
        self.df_manager = df_manager
        print("✅ CleaningToolSet initialized with a DataFrameManager.")

    # THE ONLY CHANGE IS ON THE NEXT LINE'S SIGNATURE (REMOVING *args, **kwargs)
    def get_missing_values_summary(self) -> str:
        """
        Analyzes the current DataFrame and returns a string summary of columns 
        that have missing values, including the count and percentage of missing data.
        This should be the first step to understand the data's condition.
        """
        df = self.df_manager.get_df()
        null_counts = df.isnull().sum()
        summary_df = pd.DataFrame({
            'missing_count': null_counts,
            'missing_percentage': (null_counts / len(df)) * 100
        })
        summary_df = summary_df[summary_df['missing_count'] > 0]
        
        if summary_df.empty:
            return "No missing values found in the DataFrame."
        
        return f"Missing Values Summary:\n{summary_df.to_string()}"

    # (Inside the CleaningToolSet class in cleaning_agent.py)

    def impute_column(self, column_name: str, strategy: str) -> str:
        """
        Imputes (fills) missing values in a specified column using a given strategy.
        For numeric columns, valid strategies are 'mean' or 'median'.
        For categorical/object columns, the only valid strategy is 'mode'.
        """
        df = self.df_manager.get_df()
        if column_name not in df.columns:
            return f"Error: Column '{column_name}' not found."

        try:
            if strategy == 'mean':
                fill_value = df[column_name].mean()
            elif strategy == 'median':
                fill_value = df[column_name].median()
            elif strategy == 'mode':
                fill_value = df[column_name].mode()[0]
            else:
                return f"Error: Invalid strategy '{strategy}'. Use 'mean', 'median', or 'mode'."
            
            # --- THIS IS THE ONLY CHANGE ---
            # This is a more robust way to assign the result back to the DataFrame
            df[column_name] = df[column_name].fillna(fill_value)
            
            self.df_manager.update_df(df)
            return f"Successfully imputed column '{column_name}' with strategy '{strategy}'."
        except Exception as e:
            return f"An error occurred while imputing {column_name}: {e}"
            
    def get_tools(self):
        """Returns a list of Tool objects for the agent."""
        summary_tool = StructuredTool.from_function(
            func=self.get_missing_values_summary,
            name="get_missing_values_summary",
            description="Analyzes the current DataFrame and returns a string summary of columns that have missing values."
        )
        
        impute_tool = StructuredTool.from_function(
            func=self.impute_column,
            name="impute_column",
            description="Imputes missing values in a specified column using a given strategy ('mean', 'median', or 'mode')."
        )
        
        return [summary_tool, impute_tool]

def create_cleaning_agent(tools: List):
    """Builds and returns the LangChain agent for data cleaning."""
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Data Cleaning Agent. Start by using 'get_missing_values_summary' to see what needs cleaning. Then, use the 'impute_column' tool to clean the data as requested by the user. Finally, confirm the actions taken.",
            ),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True
    )
    
    print("✅ Tool Calling Cleaning Agent created successfully.")
    return executor