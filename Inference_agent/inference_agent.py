from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from typing import List

def create_inference_agent(tools: List):
    """Builds and returns the LangChain agent for statistical inference."""
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a diligent and precise Data Inference Agent.
Your ONLY job is to perform a complete statistical analysis on a given DataFrame using the provided tools.
You MUST call ALL available tools to generate a comprehensive statistical summary.
Your final answer MUST be the complete, raw dictionary or JSON output from the tools, and nothing else. Do not add any conversational text or explanations.
Combine the outputs of all tools into a single JSON object as your final answer."""
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
    
    print("✅ Inference Agent created successfully.")
    return executor
