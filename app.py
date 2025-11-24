"""
LLM Interpretability Dashboard
Penn Claude Builder Club
"""

import streamlit as st
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from transformers import AutoModelForCausalLM, AutoTokenizer
import ollama
import time
import io

# Page config
st.set_page_config(
    page_title="LLM Interpretability Dashboard using Local Inference)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark mode compatible
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(128, 128, 128, 0.2);
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(102, 126, 234, 0.3);
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results' not in st.session_state:
    st.session_state.results = {}

# Cache model loading
@st.cache_resource
def load_hf_model(model_name="gpt2"):
    """Load any HuggingFace causal LM model"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_attentions=True,
            output_hidden_states=True,  # For Logit Lens
            trust_remote_code=True
        )
        model.eval()
        return tokenizer, model, None
    except Exception as e:
        return None, None, str(e)

def get_logit_lens(model, tokenizer, hidden_states):
    """Apply Logit Lens - decode hidden states at each layer"""
    logit_lens_results = []

    # Get the language model head (unembedding matrix)
    if hasattr(model, 'lm_head'):
        lm_head = model.lm_head
    elif hasattr(model, 'embed_out'):
        lm_head = model.embed_out
    else:
        return None

    for layer_idx, hidden in enumerate(hidden_states):
        # Apply layer norm if available (for GPT-2 style models)
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
            normed = model.transformer.ln_f(hidden)
        else:
            normed = hidden

        # Project to vocabulary
        logits = lm_head(normed)

        # Get top prediction at last position
        last_logits = logits[0, -1, :]
        probs = F.softmax(last_logits, dim=-1)
        top_prob, top_idx = torch.topk(probs, 1)
        top_token = tokenizer.decode([top_idx[0]])

        logit_lens_results.append({
            'layer': layer_idx,
            'top_token': top_token,
            'probability': top_prob[0].item(),
            'logits': last_logits.detach().cpu().numpy()
        })

    return logit_lens_results

def calculate_token_importance(attentions, tokens):
    """Calculate token importance from attention weights"""
    all_attention = np.stack([att[0].mean(axis=0) for att in attentions])
    avg_attention = all_attention.mean(axis=0)
    importance = avg_attention.sum(axis=0)
    importance = importance / importance.max()
    return importance

def run_model_analysis(model, tokenizer, prompt, top_k):
    """Run full analysis on a model"""
    inputs = tokenizer(prompt, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True, labels=inputs["input_ids"])

    # Token probabilities
    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, top_k)

    token_results = []
    for prob, idx in zip(top_probs, top_indices):
        token = tokenizer.decode([idx])
        token_results.append({
            'token': token,
            'probability': prob.item(),
            'log_prob': np.log(prob.item())
        })

    # Perplexity
    perplexity = torch.exp(outputs.loss).item()

    # Token importance
    attentions_np = [att.cpu().numpy() for att in outputs.attentions]
    importance = calculate_token_importance(attentions_np, tokens)

    # Logit Lens
    logit_lens = get_logit_lens(model, tokenizer, outputs.hidden_states)

    return {
        'tokens': tokens,
        'token_predictions': token_results,
        'attentions': attentions_np,
        'num_layers': len(outputs.attentions),
        'num_heads': outputs.attentions[0].shape[1],
        'perplexity': perplexity,
        'token_importance': importance,
        'loss': outputs.loss.item(),
        'logit_lens': logit_lens,
        'hidden_states': [h.cpu().numpy() for h in outputs.hidden_states]
    }

def fig_to_svg(fig):
    """Convert plotly figure to SVG string"""
    return fig.to_image(format="svg").decode("utf-8")

# Sidebar
with st.sidebar:
    st.header("Settings")

    st.subheader("Model Selection")

    # Custom model input
    use_custom_model = st.checkbox("Use Custom Model ID", value=False)

    if use_custom_model:
        hf_model = st.text_input(
            "HuggingFace Model ID",
            value="gpt2",
            help="Enter any HuggingFace model ID (e.g., 'EleutherAI/gpt-neo-125m', 'facebook/opt-125m')"
        )
        st.caption("Examples: gpt2, distilgpt2, EleutherAI/gpt-neo-125m, facebook/opt-125m")
    else:
        hf_model = st.selectbox(
            "Model",
            ["gpt2", "distilgpt2", "EleutherAI/gpt-neo-125m"],
            help="Select a pre-configured model"
        )

    enable_comparison = st.checkbox("Enable Model Comparison", value=False)
    if enable_comparison:
        if use_custom_model:
            comparison_model = st.text_input("Comparison Model ID", value="distilgpt2")
        else:
            comparison_model = st.selectbox("Comparison Model", ["distilgpt2", "gpt2"])

    ollama_model = st.selectbox("Ollama Model", ["llama3.2:3b", "phi3:mini"])

    st.markdown("---")
    st.subheader("Parameters")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    top_k = st.slider("Top K Tokens", 5, 25, 10)

    st.markdown("---")
    show_model_info = st.checkbox("Show Model Config", value=False)

    if show_model_info:
        model_info = {
            "model": hf_model,
            "comparison_enabled": enable_comparison,
            "top_k": top_k,
            "device": "cpu"
        }
        st.code(json.dumps(model_info, indent=2), language="json")

    st.markdown("---")
    st.markdown("**Anthropic AI**")
    st.markdown("**Penn Claude Builder Club**")

    st.markdown("---")
    st.markdown("**Albert W. Opher, IV**")
    st.markdown(
        "[Email](mailto:albert.w.opher.iv@gmail.com) | "
        "[LinkedIn](https://www.linkedin.com/in/albertopher/) | "
        "[Portfolio](https://sites.google.com/view/albert-opher-2025/home-resume)"
    )

# Header
st.title("LLM Interpretability Dashboard")
st.caption(
    "Developed as a live demonstration for the Penn Claude Builder Club (50+ attendees) "
    "showcasing ML interpretability techniques. Built with PyTorch, HuggingFace Transformers, "
    "Ollama, and Streamlit to visualize token probability distributions, attention mechanisms, "
    "logit lens analysis, perplexity metrics, and agent reasoning traces."
)
st.markdown("---")

# Prompt input
st.subheader("Analysis Prompt")
prompt = st.text_area(
    "Enter prompt:",
    value="The capital of France is",
    height=80
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    run_analysis = st.button("Run Analysis", type="primary")
with col2:
    if st.session_state.analysis_complete:
        st.success("Complete")

if run_analysis:
    with st.spinner(f"Loading {hf_model} and running analysis..."):
        try:
            tokenizer, model, error = load_hf_model(hf_model)

            if error:
                st.error(f"Failed to load model: {error}")
            else:
                primary_results = run_model_analysis(model, tokenizer, prompt, top_k)
                primary_results['prompt'] = prompt
                primary_results['model_name'] = hf_model

                comparison_results = None
                if enable_comparison:
                    tokenizer2, model2, error2 = load_hf_model(comparison_model)
                    if not error2:
                        comparison_results = run_model_analysis(model2, tokenizer2, prompt, top_k)
                        comparison_results['model_name'] = comparison_model

                st.session_state.results = primary_results
                st.session_state.comparison_results = comparison_results
                st.session_state.analysis_complete = True
                st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")

# Display results
if st.session_state.analysis_complete:
    results = st.session_state.results
    comparison_results = st.session_state.get('comparison_results', None)

    display_tokens = [t.replace('Ġ', ' ').replace('Ċ', '\\n') for t in results['tokens']]
    df = pd.DataFrame(results['token_predictions'])

    st.markdown(f"**Model:** `{results['model_name']}` | **Prompt:** `{results['prompt']}` | **Tokens:** {len(display_tokens)}")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Token Probabilities",
        "Logit Lens",
        "Perplexity",
        "Token Importance",
        "Attention",
        "Model Comparison" if comparison_results else "Agent"
    ])

    # TAB 1: Token Probabilities
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=df['token'], y=df['probability'],
                    text=df['probability'].apply(lambda x: f'{x:.1%}'),
                    textposition='outside',
                    marker=dict(color=df['probability'], colorscale='Viridis', showscale=True)
                )
            ])
            fig_bar.update_layout(
                title="Next Token Distribution",
                yaxis_tickformat='.0%', template="plotly_white", height=400
            )
            st.plotly_chart(fig_bar, use_container_width=True, key="fig_bar")

            # Export button
            if st.button("Export SVG", key="export_bar"):
                st.download_button("Download", fig_bar.to_image(format="svg"), "token_distribution.svg", "image/svg+xml")

        with col2:
            fig_pie = go.Figure(data=[
                go.Pie(labels=df['token'], values=df['probability'], hole=0.4,
                       marker=dict(colors=px.colors.sequential.Plasma))
            ])
            fig_pie.update_layout(title="Distribution", template="plotly_white", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

        top = df.iloc[0]
        entropy = -sum(p * np.log(p + 1e-10) for p in df['probability'])
        col1, col2, col3 = st.columns(3)
        col1.metric("Top Prediction", f'"{top["token"]}"')
        col2.metric("Confidence", f"{top['probability']:.1%}")
        col3.metric("Entropy", f"{entropy:.3f}")

    # TAB 2: Logit Lens
    with tab2:
        if results.get('logit_lens'):
            st.markdown("**Logit Lens** decodes hidden states at each layer to see how predictions evolve.")

            lens_data = results['logit_lens']

            # Create evolution chart
            layers = [d['layer'] for d in lens_data]
            tokens = [d['top_token'] for d in lens_data]
            probs = [d['probability'] for d in lens_data]

            fig_lens = go.Figure()

            # Probability line
            fig_lens.add_trace(go.Scatter(
                x=layers, y=probs,
                mode='lines+markers+text',
                text=tokens,
                textposition='top center',
                marker=dict(size=12, color=probs, colorscale='Viridis', showscale=True),
                line=dict(width=3, color='rgba(102, 126, 234, 0.7)'),
                hovertemplate="Layer %{x}<br>Token: %{text}<br>Prob: %{y:.2%}<extra></extra>"
            ))

            fig_lens.update_layout(
                title="Prediction Evolution Across Layers (Logit Lens)",
                xaxis_title="Layer",
                yaxis_title="Top Token Probability",
                yaxis_tickformat='.0%',
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_lens, use_container_width=True)

            if st.button("Export SVG", key="export_lens"):
                st.download_button("Download", fig_lens.to_image(format="svg"), "logit_lens.svg", "image/svg+xml")

            # Layer-by-layer table
            st.subheader("Layer-by-Layer Predictions")
            lens_df = pd.DataFrame([{
                'Layer': d['layer'],
                'Top Token': d['top_token'],
                'Probability': f"{d['probability']:.2%}"
            } for d in lens_data])
            st.dataframe(lens_df, use_container_width=True, hide_index=True)

            # Key insight
            final_token = lens_data[-1]['top_token']
            first_correct = next((d['layer'] for d in lens_data if d['top_token'] == final_token), -1)
            st.metric("Final prediction emerges at layer", first_correct if first_correct >= 0 else "N/A")
        else:
            st.warning("Logit Lens not available for this model architecture.")

    # TAB 3: Perplexity
    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            perplexity = results['perplexity']
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=perplexity,
                title={'text': "Perplexity"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 20], 'color': '#00cc96'},
                        {'range': [20, 50], 'color': '#ffa15a'},
                        {'range': [50, 100], 'color': '#ef553b'}
                    ]
                }
            ))
            fig_gauge.update_layout(height=300, template="plotly_white")
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            st.markdown("""
            | Score | Meaning |
            |-------|---------|
            | < 20 | Very predictable |
            | 20-50 | Moderate |
            | > 50 | Surprising |
            """)
            if perplexity < 20:
                st.success(f"Low perplexity ({perplexity:.2f})")
            elif perplexity < 50:
                st.info(f"Moderate ({perplexity:.2f})")
            else:
                st.warning(f"High ({perplexity:.2f})")

        col1, col2 = st.columns(2)
        col1.metric("Perplexity", f"{perplexity:.2f}")
        col2.metric("Loss", f"{results['loss']:.3f}")

    # TAB 4: Token Importance
    with tab4:
        importance = results['token_importance']

        fig_importance = go.Figure(data=[
            go.Bar(
                y=display_tokens, x=importance, orientation='h',
                marker=dict(color=importance, colorscale='Bluered', showscale=True),
                text=[f'{imp:.2f}' for imp in importance], textposition='outside'
            )
        ])
        fig_importance.update_layout(
            title="Token Importance (Attention-based)",
            xaxis_title="Importance", template="plotly_white", height=400,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig_importance, use_container_width=True)

        if st.button("Export SVG", key="export_importance"):
            st.download_button("Download", fig_importance.to_image(format="svg"), "token_importance.svg", "image/svg+xml")

        most_important_idx = np.argmax(importance)
        col1, col2 = st.columns(2)
        col1.metric("Most Important", f'"{display_tokens[most_important_idx]}"')
        col2.metric("Range", f"{importance.max() - importance.min():.3f}")

    # TAB 5: Attention
    with tab5:
        col1, col2 = st.columns(2)
        layer = col1.selectbox("Layer", range(results['num_layers']), index=0)
        head = col2.selectbox("Head", range(results['num_heads']), index=0)

        attention = results['attentions'][layer][0, head]

        fig_heat = go.Figure(data=go.Heatmap(
            z=attention, x=display_tokens, y=display_tokens,
            colorscale='RdYlBu_r', text=np.round(attention, 2),
            texttemplate='%{text}', textfont={"size": 9}
        ))
        fig_heat.update_layout(
            title=f"Attention - Layer {layer}, Head {head}",
            height=500, template="plotly_white"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        if st.button("Export SVG", key="export_attention"):
            st.download_button("Download", fig_heat.to_image(format="svg"), "attention.svg", "image/svg+xml")

        col1, col2, col3 = st.columns(3)
        col1.metric("Max", f"{attention.max():.3f}")
        col2.metric("Mean", f"{attention.mean():.3f}")
        col3.metric("Self-Attn", f"{np.diag(attention).mean():.3f}")

    # TAB 6: Model Comparison or Agent
    with tab6:
        if comparison_results:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader(results['model_name'])
                df1 = pd.DataFrame(results['token_predictions'])
                fig1 = go.Figure(data=[
                    go.Bar(x=df1['token'], y=df1['probability'],
                           marker=dict(color=df1['probability'], colorscale='Blues'),
                           text=df1['probability'].apply(lambda x: f'{x:.1%}'), textposition='outside')
                ])
                fig1.update_layout(yaxis_tickformat='.0%', template="plotly_white", height=300)
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                st.subheader(comparison_results['model_name'])
                df2 = pd.DataFrame(comparison_results['token_predictions'])
                fig2 = go.Figure(data=[
                    go.Bar(x=df2['token'], y=df2['probability'],
                           marker=dict(color=df2['probability'], colorscale='Oranges'),
                           text=df2['probability'].apply(lambda x: f'{x:.1%}'), textposition='outside')
                ])
                fig2.update_layout(yaxis_tickformat='.0%', template="plotly_white", height=300)
                st.plotly_chart(fig2, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            diff = results['perplexity'] - comparison_results['perplexity']
            col1.metric("Perplexity Diff", f"{diff:.2f}")
            top1, top2 = results['token_predictions'][0]['token'], comparison_results['token_predictions'][0]['token']
            col2.metric("Top Match", "Yes" if top1 == top2 else "No")
            overlap = len(set(t['token'] for t in results['token_predictions']) &
                         set(t['token'] for t in comparison_results['token_predictions']))
            col3.metric("Overlap", f"{overlap}/{top_k}")

        else:
            if st.button("Run Agent", type="primary"):
                with st.spinner("Reasoning..."):
                    try:
                        response = ollama.chat(
                            model=ollama_model,
                            messages=[
                                {"role": "system", "content": "You are an AI interpretability expert. Analyze prompts concisely."},
                                {"role": "user", "content": f"Analyze: '{results['prompt']}'"}
                            ],
                            options={"temperature": temperature}
                        )
                        st.markdown(response['message']['content'])
                    except Exception as e:
                        st.error(f"Error: {e}")

    # Export all results
    st.markdown("---")
    st.subheader("Export Results")

    export_data = {
        "model": results['model_name'],
        "prompt": results['prompt'],
        "perplexity": results['perplexity'],
        "loss": results['loss'],
        "top_predictions": results['token_predictions'],
        "token_importance": results['token_importance'].tolist(),
        "num_layers": results['num_layers'],
        "num_heads": results['num_heads']
    }

    if results.get('logit_lens'):
        export_data['logit_lens'] = [
            {'layer': d['layer'], 'token': d['top_token'], 'prob': d['probability']}
            for d in results['logit_lens']
        ]

    st.download_button(
        "Download JSON Report",
        json.dumps(export_data, indent=2),
        "analysis_report.json",
        "application/json"
    )

else:
    st.info("Enter a prompt and click 'Run Analysis' to begin.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>"
    "<p style='color: #888; margin-bottom: 8px;'>Penn Claude Builder Club | PyTorch | HuggingFace | Ollama</p>"
    "<p style='margin-bottom: 4px;'><strong>Albert W. Opher, IV</strong></p>"
    "<p>"
    "<a href='mailto:albert.w.opher.iv@gmail.com' style='margin: 0 10px;'>Email</a> | "
    "<a href='https://www.linkedin.com/in/albertopher/' target='_blank' style='margin: 0 10px;'>LinkedIn</a> | "
    "<a href='https://sites.google.com/view/albert-opher-2025/home-resume' target='_blank' style='margin: 0 10px;'>Portfolio</a>"
    "</p>"
    "</div>",
    unsafe_allow_html=True
)
