"""
OpenEnv Compliance Audit — Model Leaderboard Dashboard

Interactive dashboard to visualize model performance across compliance tasks.
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import json
import requests
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import os

# Configure page
st.set_page_config(
    page_title="OpenEnv Leaderboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .leaderboard-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .st-table {
        border-collapse: collapse;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("""
# ⚖️ OpenEnv Compliance Audit
## Model Performance Leaderboard

Real-time ranking of language models on 6 compliance audit tasks.
""")

# Initialize session state
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = None
if 'cache_time' not in st.session_state:
    st.session_state.cache_time = None

# Load data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_benchmark_data():
    """Load inference benchmark data from JSONL file"""
    benchmark_file = Path("model-benchmark-logs/inference_runs.jsonl")
    
    if not benchmark_file.exists():
        return pd.DataFrame()
    
    try:
        records = []
        with open(benchmark_file, 'r') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)

        # Normalize older log schema to the dashboard's expected column names.
        if 'model_name' not in df.columns and 'model' in df.columns:
            df = df.rename(columns={'model': 'model_name'})
        
        # Parse timestamps
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    except Exception as e:
        st.warning(f"Could not load benchmark data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_baseline_scores():
    """Fetch baseline scores from /baseline endpoint"""
    try:
        resp = requests.get("https://kowshik147-auditrix.hf.space/baseline", timeout=10)
        if resp.status_code == 200:
            baseline_payload = resp.json()

            # Normalize the live API response into a shape the dashboard can use directly.
            task_scores = {
                item.get('task_id'): item.get('score', 0)
                for item in baseline_payload.get('tasks', [])
                if item.get('task_id')
            }
            baseline_payload['scores'] = task_scores
            return baseline_payload
        return None
    except Exception as e:
        st.warning(f"Could not fetch baseline scores: {e}")
        return None

# Load data
benchmark_df = load_benchmark_data()
baseline_data = load_baseline_scores()

if not benchmark_df.empty and 'model_name' in benchmark_df.columns:
    discovered_models = sorted(benchmark_df['model_name'].dropna().unique().tolist())
else:
    discovered_models = []

comparison_targets = [
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "mistralai/Mistral-Large-Instruct-2407",
    "google/gemma-2-27b-it",
]

# Sidebar filters
st.sidebar.title("📊 Dashboard Controls")

if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Data source selection
data_source = st.sidebar.radio(
    "Select Data Source:",
    ["Baseline (Rule-based)", "Inference Runs (LLM)", "All Data"],
    index=2,
)

# Task filter
all_tasks = ["easy_basic_audit", "medium_mixed_audit", "hard_complex_audit", 
             "finance_sox_audit", "gdpr_privacy_audit", "data_integrity_audit"]
selected_tasks = st.sidebar.multiselect(
    "Filter by Task:",
    all_tasks,
    default=all_tasks
)

# Model filter (if inference data exists)
if discovered_models:
    selected_models = st.sidebar.multiselect(
        "Filter by Model:",
        discovered_models,
        default=discovered_models
    )
else:
    selected_models = []

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Leaderboard",
    "📈 Performance Charts",
    "📊 Detailed Results",
    "ℹ️ About"
])

# ===== TAB 1: LEADERBOARD =====
with tab1:
    st.subheader("Model Rankings by Average Score")
    
    if data_source in ["Baseline (Rule-based)", "All Data"] and baseline_data:
        st.write("### Baseline Scores (Rule-based Strategy)")
        st.caption("Baseline is a fixed rule-based agent. These scores are deterministic and do not change by LLM model.")
        
        baseline_scores = baseline_data.get('scores', {})
        baseline_df = pd.DataFrame({
            'Model': ['OpenEnv Baseline'],
            **{task: [baseline_scores.get(task, 0)] for task in selected_tasks}
        })
        
        # Calculate mean across selected tasks
        baseline_df['Mean Score'] = baseline_df[[t for t in selected_tasks if t in baseline_df.columns]].mean(axis=1)
        baseline_df['Mean Score'] = baseline_df['Mean Score'].round(4)
        
        # Sort by mean score descending
        baseline_df_sorted = baseline_df.sort_values('Mean Score', ascending=False).reset_index(drop=True)
        baseline_df_sorted.index = baseline_df_sorted.index + 1
        
        st.dataframe(
            baseline_df_sorted,
            use_container_width=True,
            height=120
        )
        
        st.info(f"✅ Baseline Mean Score: **{baseline_data.get('mean_score', 0):.4f}**")
    
    if data_source in ["Inference Runs (LLM)", "All Data"] and not benchmark_df.empty:
        st.write("### LLM Model Rankings (Inference Runs)")
        
        if len(discovered_models) <= 1:
            st.info(
                "Only one model is currently present in the benchmark log. "
                "Add more benchmark runs to compare multiple models here."
            )
        
        # Filter data
        filtered_df = benchmark_df.copy()
        if selected_tasks:
            filtered_df = filtered_df[filtered_df['task_id'].isin(selected_tasks)]
        if selected_models:
            filtered_df = filtered_df[filtered_df['model_name'].isin(selected_models)]
        
        if not filtered_df.empty:
            # Group by model and calculate statistics
            model_stats = filtered_df.groupby('model_name').agg({
                'score': ['mean', 'count', 'min', 'max', 'std'],
                'steps': 'mean',
                'success': 'mean'
            }).round(4)
            
            model_stats.columns = ['Mean Score', 'Runs', 'Min Score', 'Max Score', 'Std Dev', 'Avg Steps', 'Success Rate']
            model_stats = model_stats.sort_values('Mean Score', ascending=False)
            model_stats.index.name = 'Model'
            model_stats = model_stats.reset_index()
            model_stats.index = model_stats.index + 1
            
            # Color coding
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.dataframe(
                    model_stats,
                    use_container_width=True,
                    height=300
                )
            with col2:
                st.metric("Top Model", model_stats.iloc[0]['Model'] if len(model_stats) > 0 else "N/A")
                st.metric("Top Score", f"{model_stats.iloc[0]['Mean Score']:.4f}" if len(model_stats) > 0 else "N/A")
            with col3:
                st.metric("Total Runs", model_stats['Runs'].sum())
                st.metric("Models", len(model_stats))
        else:
            st.warning("No inference data available. Run inference.py to populate data.")

        if discovered_models:
            st.write("### Discovered Models")
            model_summary = pd.DataFrame({
                "Model": discovered_models,
                "Runs": [int((benchmark_df['model_name'] == model).sum()) for model in discovered_models]
            })
            st.dataframe(model_summary, use_container_width=True, height=140)

        st.write("### Suggested Comparison Targets")
        comparison_df = pd.DataFrame({
            "Model": comparison_targets,
            "Status": ["Benchmarked" if model in discovered_models else "Not benchmarked yet" for model in comparison_targets],
            "Runs": [int((benchmark_df['model_name'] == model).sum()) if model in discovered_models else 0 for model in comparison_targets],
        })
        st.dataframe(comparison_df, use_container_width=True, height=180)
    
    if data_source == "All Data" and benchmark_df.empty and not baseline_data:
        st.warning("No data available. Please run inference or check baseline endpoint.")

# ===== TAB 2: PERFORMANCE CHARTS =====
with tab2:
    col1, col2 = st.columns(2)
    
    # Chart 1: Mean score by task
    if baseline_data and "Baseline" in data_source:
        with col1:
            st.subheader("Score by Task (Baseline)")
            baseline_scores = baseline_data.get('scores', {})
            task_scores = {k: v for k, v in baseline_scores.items() if k in selected_tasks}
            
            if task_scores:
                chart_df = pd.DataFrame({
                    'Task': list(task_scores.keys()),
                    'Score': list(task_scores.values())
                })
                
                fig = px.bar(chart_df, x='Task', y='Score', 
                            color='Score',
                            color_continuous_scale='Viridis',
                            title="Baseline Performance per Task",
                            height=400)
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    # Chart 2: Model comparison
    if not benchmark_df.empty and "Inference" in data_source:
        with col2:
            st.subheader("Score Distribution by Model")
            filtered_df = benchmark_df.copy()
            if selected_tasks:
                filtered_df = filtered_df[filtered_df['task_id'].isin(selected_tasks)]
            if selected_models:
                filtered_df = filtered_df[filtered_df['model_name'].isin(selected_models)]
            
            if not filtered_df.empty:
                fig = px.box(filtered_df, x='model_name', y='score',
                           title="Score Distribution Across Runs",
                           labels={'model_name': 'Model', 'score': 'Score'},
                           height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    # Chart 3: Performance over time
    if not benchmark_df.empty and 'timestamp' in benchmark_df.columns and "Inference" in data_source:
        st.subheader("Performance Trend Over Time")
        filtered_df = benchmark_df.copy()
        if selected_tasks:
            filtered_df = filtered_df[filtered_df['task_id'].isin(selected_tasks)]
        if selected_models:
            filtered_df = filtered_df[filtered_df['model_name'].isin(selected_models)]
        
        if not filtered_df.empty:
            fig = px.line(filtered_df, x='timestamp', y='score', 
                         color='model_name',
                         title="Score Trend Over Time",
                         labels={'timestamp': 'Time', 'score': 'Score', 'model_name': 'Model'},
                         height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Chart 4: Task difficulty comparison
    if not benchmark_df.empty and "Inference" in data_source:
        st.subheader("Performance by Task Difficulty")
        filtered_df = benchmark_df.copy()
        if selected_models:
            filtered_df = filtered_df[filtered_df['model_name'].isin(selected_models)]
        
        if not filtered_df.empty:
            fig = px.scatter(filtered_df, x='steps', y='score', 
                           color='task_id', size='steps',
                           title="Steps vs Score (colored by Task)",
                           labels={'steps': 'Steps Taken', 'score': 'Final Score', 'task_id': 'Task'},
                           height=400)
            st.plotly_chart(fig, use_container_width=True)

# ===== TAB 3: DETAILED RESULTS =====
with tab3:
    st.subheader("Detailed Inference Run Results")
    
    if not benchmark_df.empty and "Inference" in data_source:
        # Filter data
        filtered_df = benchmark_df.copy()
        if selected_tasks:
            filtered_df = filtered_df[filtered_df['task_id'].isin(selected_tasks)]
        if selected_models:
            filtered_df = filtered_df[filtered_df['model_name'].isin(selected_models)]
        
        # Display options
        col1, col2, col3 = st.columns(3)
        with col1:
            sort_by = st.selectbox("Sort by:", ["score", "steps", "timestamp"])
        with col2:
            sort_order = st.radio("Order:", ["Descending", "Ascending"])
        with col3:
            rows_to_show = st.slider("Rows to display:", 5, 100, 20)
        
        # Sort
        ascending = sort_order == "Ascending"
        filtered_df_sorted = filtered_df.sort_values(sort_by, ascending=ascending).head(rows_to_show)
        
        # Display table
        display_cols = ['model_name', 'task_id', 'score', 'steps', 'success', 'timestamp']
        display_cols = [c for c in display_cols if c in filtered_df_sorted.columns]
        
        st.dataframe(
            filtered_df_sorted[display_cols],
            use_container_width=True,
            height=400
        )
        
        # Export option
        csv = filtered_df_sorted.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"leaderboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No inference data available. Run `python inference.py` to generate benchmark results.")

# ===== TAB 4: ABOUT =====
with tab4:
    st.markdown("""
    ## 📖 About OpenEnv Compliance Audit
    
    **OpenEnv** is an interactive compliance audit environment for evaluating AI agents on real-world 
    HR and regulatory compliance tasks.
    
    ### 🎯 6 Tasks Across 3 Difficulty Levels:
    
    - **🟢 Easy (2):** Basic HR compliance with 2 rules
    - **🟡 Medium (3):** Mixed HR & payroll with 4 rules
    - **🔴 Hard (1+):** Complex scenarios with all regulatory domains
    
    ### 📊 Dashboard Features:
    
    - **Leaderboard:** Model rankings by average score
    - **Performance Charts:** Visualize trends and distributions
    - **Detailed Results:** Filter and sort inference runs
    - **Comparison Targets:** Suggested additional models to benchmark
    - **Real-time Data:** Automatic updates from `model-benchmark-logs/`
    
    ### 🔗 Links:
    
    - [GitHub Repository](https://github.com/Kowshikv07/Auditrix)
    - [HF Space Demo](https://huggingface.co/spaces/kowshik147/openenv-ticket-triage)
    - [OpenEnv Framework](https://github.com/huggingface/openenv)
    
    ### 💡 How to Use:
    
    1. **Run inference:** `python inference.py`
    2. **Open dashboard:** `streamlit run dashboard.py`
    3. **View results:** Leaderboard updates automatically
    
    ---
    
    **Built for:** OpenEnv Hackathon 2025  
    **Team:** Kowshik & Team  
    **Status:** Production Ready ⚡
    """)
    
    # Stats
    st.divider()
    st.subheader("📈 Dashboard Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tasks", 6)
    with col2:
        st.metric("Difficulty Levels", 3)
    with col3:
        if not benchmark_df.empty:
            st.metric("Inference Runs", len(benchmark_df))
        else:
            st.metric("Inference Runs", 0)
    with col4:
        if baseline_data:
            st.metric("Baseline Mean", f"{baseline_data.get('mean_score', 0):.4f}")
        else:
            st.metric("Baseline Mean", "N/A")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px;'>
    ⚖️ OpenEnv Compliance Audit | Last Updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """ UTC
</div>
""", unsafe_allow_html=True)
