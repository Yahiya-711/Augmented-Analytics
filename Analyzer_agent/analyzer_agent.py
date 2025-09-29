from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_analyzer_chain():
    """
    Builds and returns a LangChain chain that takes a statistical summary
    and generates a natural language report.
    """
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
    
    # This detailed prompt is the "brain" of our analyzer.
    # It guides the LLM on how to structure its thoughts and what to focus on.
    prompt = ChatPromptTemplate.from_template(
        """You are an expert data analyst. Your job is to take a JSON object containing a statistical summary of a dataset and write a clear, concise, and insightful report for a business audience.

        Here is the statistical summary you need to analyze:
        {stats_json}

        Please structure your report with the following sections, using markdown for formatting:

        ### 1. Executive Summary
        A brief, high-level overview of the most critical findings.

        ### 2. Key Statistical Findings
        - Describe the main characteristics of the numerical data (e.g., age, salary). Mention the average, median, and range.
        - Discuss the distribution of the categorical data (e.g., city).
        
        ### 3. Data Quality & Outliers
        - Point out any potential data quality issues based on the outlier detection. Mention which columns have outliers and what this might imply (e.g., data entry errors, or genuinely exceptional cases).

        ### 4. Actionable Business Insights
        - Based on all the information, provide 1-2 concrete insights that a business could act on. For example, "The significant salary outlier could represent a high-value client or a senior employee, warranting further investigation." or "The dominance of 'New York' in the city data suggests this is a key market."

        Generate the report based on the provided JSON data.
        """
    )
    
    parser = StrOutputParser()
    
    # A "chain" simply connects the components in sequence.
    chain = prompt | llm | parser
    
    print("✅ Analyzer Agent (Chain) created successfully.")
    return chain
