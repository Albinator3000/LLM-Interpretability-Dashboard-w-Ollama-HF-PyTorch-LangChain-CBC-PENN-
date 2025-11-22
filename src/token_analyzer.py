import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict

class TokenAnalyzer:
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.model.eval()

    def get_top_k_predictions(self, prompt: str, top_k: int = 10) -> pd.DataFrame:
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]

        probs = torch.softmax(logits, dim=-1)

        # Get top-k
        top_probs, top_indices = torch.topk(probs, top_k)
        
        # Decode
        results = []
        for prob, idx in zip(top_probs, top_indices):
            token = self.tokenizer.decode([idx])
            results.append({
                'token': token,
                'probability': prob.item(),
                'log_probability': torch.log(prob).item()
            })
        
        return pd.DataFrame(results)
    
    def visualize_probabilities(self, df: pd.DataFrame, title: str = "Top Token Predictions") -> go.Figure:
        """Create bar chart"""
        
        fig = go.Figure(data=[
            go.Bar(
                x=df['token'],
                y=df['probability'],
                text=df['probability'].apply(lambda x: f'{x:.3%}'),
                textposition='auto',
                marker_color='rgb(212, 145, 126)'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Token",
            yaxis_title="Probability",
            yaxis_tickformat='.0%',
            template="plotly_white",
            height=400
        )
        
        return fig