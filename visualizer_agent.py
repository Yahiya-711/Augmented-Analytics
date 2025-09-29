import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Any, Union
import streamlit as st

class VisualizerToolSet:
    """A class that holds a DataFrame and exposes visualization methods as tools."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        print("✅ VisualizerToolSet initialized with a DataFrame.")

    def plot_histogram(self, column_name: str) -> str:
        """
        Generates and stores an interactive histogram for a single numerical column.
        
        Args:
            column_name (str): The name of the numerical column to plot.
        """
        try:
            if column_name not in self.df.columns:
                return f"Error: Column '{column_name}' not found in the data."
                
            # Check if column is numerical
            if not pd.api.types.is_numeric_dtype(self.df[column_name]):
                return f"Error: Column '{column_name}' is not numerical. Histograms require numerical data."
            
            fig = px.histogram(self.df, x=column_name, title=f"Distribution of {column_name}")
            
            # Store the chart in Streamlit session state directly
            if 'current_chart' not in st.session_state:
                st.session_state.current_chart = None
            st.session_state.current_chart = fig
            
            return f"✅ Successfully created histogram for '{column_name}'"
            
        except Exception as e:
            return f"Error creating histogram: {str(e)}"

    def plot_bar_chart(self, column_name: str) -> str:
        """
        Generates and stores an interactive bar chart for a single categorical column.

        Args:
            column_name (str): The name of the categorical column to plot.
        """
        try:
            if column_name not in self.df.columns:
                return f"Error: Column '{column_name}' not found in the data."
            
            counts = self.df[column_name].value_counts().reset_index()
            counts.columns = [column_name, 'count']
            fig = px.bar(counts, x=column_name, y='count', title=f"Value Counts of {column_name}")
            
            # Store the chart in Streamlit session state directly
            if 'current_chart' not in st.session_state:
                st.session_state.current_chart = None
            st.session_state.current_chart = fig
                
            return f"✅ Successfully created bar chart for '{column_name}'"
            
        except Exception as e:
            return f"Error creating bar chart: {str(e)}"

    def plot_scatter(self, x_column: str, y_column: str) -> str:
        """
        Generates and stores an interactive scatter plot for two numerical columns.

        Args:
            x_column (str): The column for the x-axis.
            y_column (str): The column for the y-axis.
        """
        try:
            # Validate columns exist
            missing_cols = [col for col in [x_column, y_column] if col not in self.df.columns]
            if missing_cols:
                available_cols = list(self.df.columns)
                return f"Error: Column(s) {missing_cols} not found. Available columns: {available_cols}"
            
            # Check if columns are numerical
            non_numeric = []
            for col in [x_column, y_column]:
                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    non_numeric.append(col)
            
            if non_numeric:
                return f"Error: Column(s) {non_numeric} are not numerical. Scatter plots require numerical data."
            
            fig = px.scatter(self.df, x=x_column, y=y_column, 
                           title=f"Relationship between {x_column} and {y_column}")
            
            # Store the chart in Streamlit session state directly
            if 'current_chart' not in st.session_state:
                st.session_state.current_chart = None
            st.session_state.current_chart = fig
            
            return f"✅ Successfully created scatter plot of '{x_column}' vs '{y_column}'"
            
        except Exception as e:
            return f"Error creating scatter plot: {str(e)}"

    def get_tools(self):
        """Returns a list of Tool objects for the agent."""
        return [
            StructuredTool.from_function(
                func=self.plot_histogram, 
                name="plot_histogram", 
                description="Creates a histogram for a numerical column. Returns success message."
            ),
            StructuredTool.from_function(
                func=self.plot_bar_chart, 
                name="plot_bar_chart", 
                description="Creates a bar chart for a categorical column. Returns success message."
            ),
            StructuredTool.from_function(
                func=self.plot_scatter, 
                name="plot_scatter", 
                description="Creates a scatter plot for two numerical columns. Returns success message."
            )
        ]

def create_visualizer_agent(tools: List):
    """Builds and returns the LangChain agent for data visualization."""
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are an expert Data Visualization Agent. 

Your job is to create plots based on user requests using the provided tools.

IMPORTANT INSTRUCTIONS:
1. Use the appropriate tool based on the user's request
2. For scatter plots, use plot_scatter with two numerical columns
3. For histograms, use plot_histogram with one numerical column  
4. For bar charts, use plot_bar_chart with one categorical column
5. The tools will return a success message when charts are created
6. Simply relay the success message back to the user

The actual chart visualization will be handled automatically by the system."""),
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
    print("✅ Visualizer Agent created successfully.")
    return executor