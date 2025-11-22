"""
LLM Interpretability Dashboard
Main Streamlit Application
"""

import streamlit as st
import json
from src.token_analyzer import TokenAnalyzer
from src.attention_visualizer import AttentionVisualizer
from src.agent_tracer import AgentTracer
from src.utils import load_prompts as load_demo_prompts

# Page config
st.set_page_config(
    page_title="LLM Interpretability Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("LLM Interpretability Dashboard")
st.markdown("---")

# Load demo prompts
demo_prompts = load_demo_prompts()

# Initialize models (cached)
@st.cache_resource
def load_token_analyzer():
    return TokenAnalyzer()

@st.cache_resource
def load_attention_visualizer():
    return AttentionVisualizer()

@st.cache_resource
def load_agent_tracer():
    return AgentTracer()

# Sidebar
with st.sidebar:
    st.header("Settings")

    st.subheader("Models")
    hf_model = st.selectbox(
        "HuggingFace Model",
        ["gpt2", "distilgpt2"],
        help="Model for token/attention"
    )

    ollama_model = st.selectbox(
        "Ollama Model",
        ["llama3.2:3b", "phi3:mini"],
        help="Model for agents"
    )

    st.markdown("---")
    st.subheader("Parameters")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.slider("Max Tokens", 50, 2000, 512, 50)

    st.markdown("---")
    show_model_info = st.checkbox("Show Model Info", value=False)

    if show_model_info:
        model_info = {
            "huggingface_model": hf_model,
            "ollama_model": ollama_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "framework": "PyTorch",
            "device": "auto"
        }
        st.code(json.dumps(model_info, indent=2), language="json")

    st.markdown("---")
    st.subheader("About")
    st.info(
        "Three interpretability techniques:\n\n"
        "1. Token Probabilities\n"
        "2. Attention Patterns\n"
        "3. Agent Reasoning"
    )

    st.markdown("---")
    st.markdown("**Penn Claude Builder Club**")

# Tabs
tab1, tab2, tab3 = st.tabs([
    "Token Probabilities",
    "Attention Patterns",
    "Agent Reasoning"
])

# TAB 1: Token Probabilities
with tab1:
    st.header("Token Probability Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        prompt = st.text_area(
            "Enter prompt:",
            value=demo_prompts['token_analysis'][0],
            height=100,
            key="token_prompt"
        )
    
    with col2:
        st.markdown("**Examples:**")
        for example in demo_prompts['token_analysis'][:3]:
            if st.button(example, key=f"tok_{example}"):
                st.session_state.token_prompt = example
                st.rerun()
        
        top_k = st.slider("Top K", 5, 20, 10)
    
    if st.button("Analyze", type="primary", key="analyze_tokens"):
        with st.spinner("Analyzing..."):
            try:
                analyzer = load_token_analyzer()
                predictions = analyzer.get_top_k_predictions(prompt, top_k=top_k)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Predictions")
                    fig = analyzer.visualize_probabilities(predictions)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("Details")
                    st.dataframe(
                        predictions[['token', 'probability']].style.format({
                            'probability': '{:.2%}'
                        }),
                        height=400,
                        use_container_width=True
                    )
                
                # Insights
                st.subheader("Insights")
                top = predictions.iloc[0]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Top Token", f'"{top["token"]}"')
                with col2:
                    st.metric("Confidence", f"{top['probability']:.1%}")
                with col3:
                    st.metric("Uncertainty", f"{1-top['probability']:.1%}")

                if top['probability'] > 0.5:
                    st.success("High confidence")
                elif top['probability'] > 0.3:
                    st.info("Moderate confidence")
                else:
                    st.warning("Low confidence")
                    
            except Exception as e:
                st.error(f"Error: {e}")

# TAB 2: Attention
with tab2:
    st.header("Attention Pattern Visualization")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        text = st.text_area(
            "Enter text:",
            value=demo_prompts['attention'][0],
            height=100,
            key="attention_text"
        )
    
    with col2:
        st.markdown("**Examples:**")
        for example in demo_prompts['attention']:
            if st.button(example, key=f"att_{example}"):
                st.session_state.attention_text = example
                st.rerun()
    
    try:
        visualizer = load_attention_visualizer()
        info = visualizer.get_model_info()
        
        col1, col2 = st.columns(2)
        with col1:
            layer = st.slider("Layer", 0, info['num_layers']-1, 0)
        with col2:
            head = st.slider("Head", 0, info['num_heads']-1, 0)
    except:
        layer = 0
        head = 0
    
    if st.button("Visualize", type="primary", key="viz_attention"):
        with st.spinner("Extracting attention..."):
            try:
                visualizer = load_attention_visualizer()
                attention, tokens = visualizer.get_attention_weights(text, layer, head)
                
                st.subheader(f"Layer {layer}, Head {head}")
                fig = visualizer.visualize_attention_heatmap(attention, tokens)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("How to Read")
                st.markdown("""
                - **Rows**: Token asking
                - **Columns**: Tokens looked at
                - **Red = Strong attention**
                - **Diagonal = Self-attention**
                """)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tokens", len(tokens))
                with col2:
                    st.metric("Avg", f"{attention.mean():.3f}")
                with col3:
                    st.metric("Max", f"{attention.max():.3f}")
                
            except Exception as e:
                st.error(f"Error: {e}")

# TAB 3: Agent
with tab3:
    st.header("Agent Decision Tracing")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        task = st.text_area(
            "Enter task:",
            value=demo_prompts['agent_tasks'][0],
            height=100,
            key="agent_task"
        )
    
    with col2:
        st.markdown("**Examples:**")
        for example in demo_prompts['agent_tasks'][:3]:
            if st.button(example, key=f"agt_{example}"):
                st.session_state.agent_task = example
                st.rerun()
    
    if st.button("Run Agent", type="primary", key="run_agent"):
        with st.spinner("Agent thinking..."):
            try:
                tracer = load_agent_tracer()
                result = tracer.run_agent(task)
                
                if result['success']:
                    st.success("Complete")
                    st.subheader("Answer")
                    st.info(result['output'])

                    st.subheader("Trace")
                    formatted = tracer.format_trace_for_display(result['steps'])
                    
                    for step in formatted:
                        with st.expander(f"Step {step['step_num']}: {step['type']}", expanded=True):
                            if step['type'] == 'Reasoning':
                                st.markdown("**Thought:**")
                                st.text(step['content'])
                                st.markdown(f"**Tool:** `{step.get('tool', '')}`")
                                st.markdown(f"**Input:** `{step.get('input', '')}`")
                            elif step['type'] == 'Observation':
                                st.markdown("**Output:**")
                                st.code(step['content'])
                            else:
                                st.markdown(step['content'])

                    st.subheader("Stats")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Steps", len(formatted))
                    with col2:
                        actions = [s for s in formatted if 'Reasoning' in s['type']]
                        st.metric("Actions", len(actions))
                    with col3:
                        obs = [s for s in formatted if 'Observation' in s['type']]
                        st.metric("Observations", len(obs))
                else:
                    st.error("Error")
                    st.error(result.get('error'))
                
            except Exception as e:
                st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Penn Claude Builder Club | PyTorch • HuggingFace • Ollama • LangChain • Streamlit"
    "</div>",
    unsafe_allow_html=True
)