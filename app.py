import streamlit as st
import pandas as pd
import sys
import os
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import Orchestrator
from Visualizer_agent.visualizer_agent import VisualizerToolSet, create_visualizer_agent

st.set_page_config(layout="wide", page_title="Gen AI-Powered Data Analysis")
st.title("🤖 Gen AI-Powered Data Analysis")

# Initialize session state
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_chart" not in st.session_state:
    st.session_state.current_chart = None

# Main app logic
uploaded_file = st.file_uploader("Upload your CSV file to begin analysis", type=["csv"])

if uploaded_file is not None and not st.session_state.analysis_complete:
    with st.spinner("🚀 Starting analysis... This may take a moment."):
        try:
            df = pd.read_csv(uploaded_file)
            orchestrator = Orchestrator()
            insights, cleaned_df = orchestrator.run_pipeline(df)

            st.session_state.cleaned_df = cleaned_df
            st.session_state.insights = insights
            st.session_state.orchestrator = orchestrator  # Store orchestrator in session state
            vis_tool_set = VisualizerToolSet(cleaned_df)
            st.session_state.visualizer_agent = create_visualizer_agent(vis_tool_set.get_tools())
            st.session_state.analysis_complete = True
            
        except Exception as e:
            st.error(f"An error occurred during the analysis pipeline: {e}")
            st.exception(e)

if st.session_state.analysis_complete:
    st.header("📊 Automated Analysis Report")
    st.markdown(st.session_state.insights)
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["💬 Data Visualizer", "🔬 What-If Analysis"])
    
    # Tab 1: Visualizer Agent
    with tab1:
        st.header("Chat with the Visualizer Agent")
        
        # Display all messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if isinstance(message["content"], dict) and "chart" in message["content"]:
                    # Display stored chart
                    st.plotly_chart(message["content"]["chart"], use_container_width=True)
                    if "text" in message["content"]:
                        st.markdown(message["content"]["text"])
                else:
                    st.markdown(message["content"])

        # Handle new chat input
        if prompt := st.chat_input("e.g., 'Create a scatter plot of Age vs Salary'"):
            # Clear any previous chart
            st.session_state.current_chart = None
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display new assistant response
            with st.chat_message("assistant"):
                with st.spinner("Generating chart..."):
                    # Get the agent result
                    result = st.session_state.visualizer_agent.invoke({"input": prompt})
                    
                    # Get the text response
                    text_response = result.get('output', 'Task completed.')
                    
                    # Check if a chart was created and stored in session state
                    if st.session_state.current_chart is not None:
                        # Display the chart
                        st.plotly_chart(st.session_state.current_chart, use_container_width=True)
                        
                        # Display success message
                        st.success(text_response)
                        
                        # Store both chart and text in message history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": {
                                "chart": st.session_state.current_chart,
                                "text": text_response
                            }
                        })
                        
                        # Clear the current chart for next interaction
                        st.session_state.current_chart = None
                    else:
                        # No chart was created, just display text
                        st.markdown(text_response)
                        st.session_state.messages.append({"role": "assistant", "content": text_response})
    
    # Tab 2: What-If Analysis
    with tab2:
        st.header("🔬 What-If Business Scenario")
        st.markdown("Use the controls below to model a change and see its impact on the analysis.")

        # Get only the numerical columns for what-if analysis
        numerical_columns = st.session_state.cleaned_df.select_dtypes(include=['number']).columns.tolist()
        
        if not numerical_columns:
            st.warning("No numerical columns found in the dataset for what-if analysis.")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_column = st.selectbox("Select a column to modify:", numerical_columns)
            
            with col2:
                change_type = st.selectbox("Select the type of change:", 
                                         ['Percentage Increase', 'Percentage Decrease', 'Set to Value'])
            
            with col3:
                if "Percentage" in change_type:
                    change_value = st.number_input("Enter the percentage %:", min_value=0.0, step=1.0, value=10.0)
                else:
                    original_mean = st.session_state.cleaned_df[selected_column].mean()
                    change_value = st.number_input("Enter the new value:", step=1.0, value=float(original_mean))
            
            # Show current statistics before running scenario
            st.markdown("### Current Statistics")
            current_stats = st.session_state.cleaned_df[selected_column].describe()
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Current Mean", f"{current_stats['mean']:.2f}")
            with col_b:
                st.metric("Current Median", f"{current_stats['50%']:.2f}")
            with col_c:
                st.metric("Current Std Dev", f"{current_stats['std']:.2f}")
                
            if st.button("🚀 Run Business Scenario", type="primary"):
                modifications = {
                    "column": selected_column,
                    "change_type": change_type,
                    "value": change_value
                }
                
                with st.spinner("🧪 Running scenario..."):
                    try:
                        scenario_report = st.session_state.orchestrator.run_what_if_scenario(
                            df=st.session_state.cleaned_df,
                            modifications=modifications
                        )
                        
                        st.markdown("### 📊 Scenario Results")
                        st.markdown(scenario_report)
                        
                        # Show the impact summary
                        st.markdown("### 📈 Impact Summary")
                        modified_df = st.session_state.cleaned_df.copy()
                        col = modifications['column']
                        change_type = modifications['change_type']
                        val = modifications['value']
                        
                        # Apply same modification to show before/after
                        if change_type == 'Percentage Increase':
                            modified_df[col] *= (1 + val / 100)
                            impact_text = f"**{val}% increase** in {col}"
                        elif change_type == 'Percentage Decrease':
                            modified_df[col] *= (1 - val / 100)
                            impact_text = f"**{val}% decrease** in {col}"
                        elif change_type == 'Set to Value':
                            modified_df[col] = val
                            impact_text = f"**Set {col} to {val}**"
                        
                        new_stats = modified_df[col].describe()
                        
                        st.markdown(f"**Change Applied:** {impact_text}")
                        
                        # Show before/after comparison
                        col_x, col_y, col_z = st.columns(3)
                        
                        with col_x:
                            st.metric(
                                "New Mean", 
                                f"{new_stats['mean']:.2f}",
                                delta=f"{new_stats['mean'] - current_stats['mean']:.2f}"
                            )
                        with col_y:
                            st.metric(
                                "New Median", 
                                f"{new_stats['50%']:.2f}",
                                delta=f"{new_stats['50%'] - current_stats['50%']:.2f}"
                            )
                        with col_z:
                            st.metric(
                                "New Std Dev", 
                                f"{new_stats['std']:.2f}",
                                delta=f"{new_stats['std'] - current_stats['std']:.2f}"
                            )
                        
                    except Exception as e:
                        st.error(f"Error running scenario: {str(e)}")
                        st.exception(e)