import pandas as pd
import sys
import os
from typing import Tuple, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the new utility functions
from utils.cleaner import clean_dataframe
from utils.profiler import profile_dataframe
from Analyzer_agent.analyzer_agent import create_analyzer_chain

class Orchestrator:
    """Manages the end-to-end data analysis pipeline."""

    def run_pipeline(self, df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
        """Executes the full Clean -> Profile -> Analyze pipeline."""
        print("🚀 Starting pipeline...")

        # 1. Clean the DataFrame
        cleaned_df = clean_dataframe(df)

        # 2. Profile the DataFrame
        stats_json = profile_dataframe(cleaned_df)

        # 3. Analyze the results with the AI agent
        print("\n--- 🧠 Invoking Analyzer Agent ---")
        analyzer_chain = create_analyzer_chain()
        insights_report = analyzer_chain.invoke({"stats_json": stats_json})
        print("--- ✅ Analysis Complete ---")

        return insights_report, cleaned_df

    def run_what_if_scenario(self, df: pd.DataFrame, modifications: Dict[str, Any]) -> str:
        """
        Runs a what-if scenario locally on a copy of the data based on direct inputs.
        
        Args:
            df: Original cleaned DataFrame
            modifications: Dict containing column, change_type, and value
            
        Returns:
            Formatted analysis report of the what-if scenario
        """
        print(f"\n--- 🚀 Starting What-If Scenario: {modifications} ---")
        
        try:
            # Create a copy to avoid modifying original data
            modified_df = df.copy()
            original_df = df.copy()
            
            # Unpack the modification instructions
            col = modifications['column']
            change_type = modifications['change_type']
            val = modifications['value']
            
            # Validate column exists
            if col not in modified_df.columns:
                return f"❌ Error: Column '{col}' not found in the dataset."
            
            # Validate column is numerical
            if not pd.api.types.is_numeric_dtype(modified_df[col]):
                return f"❌ Error: Column '{col}' is not numerical. What-if analysis requires numerical columns."
            
            # Store original statistics for comparison
            original_stats = {
                'mean': original_df[col].mean(),
                'median': original_df[col].median(),
                'std': original_df[col].std(),
                'min': original_df[col].min(),
                'max': original_df[col].max()
            }
            
            # Apply the modification based on change type
            if change_type == 'Percentage Increase':
                modified_df[col] = modified_df[col] * (1 + val / 100)
                change_description = f"increased all {col} values by {val}%"
            elif change_type == 'Percentage Decrease':
                modified_df[col] = modified_df[col] * (1 - val / 100)
                change_description = f"decreased all {col} values by {val}%"
            elif change_type == 'Set to Value':
                modified_df[col] = val
                change_description = f"set all {col} values to {val}"
            else:
                return f"❌ Error: Unknown change type '{change_type}'"
            
            print(f"--- ✅ Applied modification: {change_description} ---")
            
            # Calculate new statistics
            new_stats = {
                'mean': modified_df[col].mean(),
                'median': modified_df[col].median(),
                'std': modified_df[col].std(),
                'min': modified_df[col].min(),
                'max': modified_df[col].max()
            }
            
            # Calculate changes
            stats_changes = {}
            for stat_name in original_stats:
                if original_stats[stat_name] != 0:  # Avoid division by zero
                    percentage_change = ((new_stats[stat_name] - original_stats[stat_name]) / abs(original_stats[stat_name])) * 100
                    stats_changes[stat_name] = percentage_change
                else:
                    stats_changes[stat_name] = 0
            
            print("--- 🧮 Calculating impact on other variables ---")
            
            # Analyze impact on other numerical columns (simple correlation analysis)
            numerical_cols = modified_df.select_dtypes(include=['number']).columns.tolist()
            impact_analysis = {}
            
            for other_col in numerical_cols:
                if other_col != col:
                    # Calculate correlation between modified column and other columns
                    correlation = original_df[col].corr(original_df[other_col])
                    
                    if abs(correlation) > 0.1:  # Only report if correlation is meaningful
                        # Estimate impact based on correlation and change
                        if change_type in ['Percentage Increase', 'Percentage Decrease']:
                            estimated_impact = correlation * val * (1 if change_type == 'Percentage Increase' else -1)
                        else:
                            # For "Set to Value", calculate percentage change first
                            percentage_change = ((val - original_stats['mean']) / original_stats['mean']) * 100
                            estimated_impact = correlation * percentage_change
                        
                        impact_analysis[other_col] = {
                            'correlation': correlation,
                            'estimated_impact': estimated_impact
                        }
            
            # Re-run the Profiler and Analyzer on the MODIFIED data for comprehensive analysis
            print("--- 📊 Re-analyzing modified dataset ---")
            stats_json = profile_dataframe(modified_df)
            analyzer_chain = create_analyzer_chain()
            scenario_report = analyzer_chain.invoke({"stats_json": stats_json})
            
            # Create comprehensive what-if report
            what_if_report = f"""## 🎯 What-If Scenario Analysis Results

### 📋 Scenario Details
**Modification Applied:** {change_description}

### 📊 Direct Impact on {col}

| Statistic | Original | New | Change | % Change |
|-----------|----------|-----|--------|----------|
| **Mean** | {original_stats['mean']:.2f} | {new_stats['mean']:.2f} | {new_stats['mean'] - original_stats['mean']:.2f} | {stats_changes['mean']:.1f}% |
| **Median** | {original_stats['median']:.2f} | {new_stats['median']:.2f} | {new_stats['median'] - original_stats['median']:.2f} | {stats_changes['median']:.1f}% |
| **Std Dev** | {original_stats['std']:.2f} | {new_stats['std']:.2f} | {new_stats['std'] - original_stats['std']:.2f} | {stats_changes['std']:.1f}% |
| **Min** | {original_stats['min']:.2f} | {new_stats['min']:.2f} | {new_stats['min'] - original_stats['min']:.2f} | {stats_changes['min']:.1f}% |
| **Max** | {original_stats['max']:.2f} | {new_stats['max']:.2f} | {new_stats['max'] - original_stats['max']:.2f} | {stats_changes['max']:.1f}% |

### 🔗 Estimated Impact on Related Variables
"""

            if impact_analysis:
                for other_col, impact_data in impact_analysis.items():
                    correlation = impact_data['correlation']
                    estimated_impact = impact_data['estimated_impact']
                    correlation_strength = "Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.3 else "Weak"
                    direction = "positive" if correlation > 0 else "negative"
                    
                    what_if_report += f"""
**{other_col}:**
- Correlation with {col}: {correlation:.3f} ({correlation_strength} {direction})
- Estimated impact: {estimated_impact:.1f}% change
"""
            else:
                what_if_report += "\nNo significant correlations found with other numerical variables."
            
            what_if_report += f"""

### 🧠 AI Analysis of Modified Dataset

{scenario_report}

---
*💡 This analysis shows potential impacts based on statistical relationships in your data. Actual business results may vary based on external factors not captured in this dataset.*
"""
            
            print("--- ✅ What-If Analysis Complete ---")
            return what_if_report
            
        except Exception as e:
            error_message = f"❌ Error during what-if analysis: {str(e)}"
            print(error_message)
            return error_message
