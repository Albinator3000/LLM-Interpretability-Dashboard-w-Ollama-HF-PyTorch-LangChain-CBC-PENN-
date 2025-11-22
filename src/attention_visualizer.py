"""
Attention Pattern Visualizer
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import plotly.graph_objects as go
import numpy as np
from typing import Tuple, List

class AttentionVisualizer:
    """Visualize attention patterns"""

    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load model with attention"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            output_attentions=True,
            trust_remote_code=True
        )
        self.model.eval()
    
    def get_attention_weights(self, text: str, layer: int = 0, head: int = 0) -> Tuple[np.ndarray, List[str]]:
        """Extract attention weights"""
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        # Extract attention
        attention = outputs.attentions[layer][0, head].cpu().numpy()
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        return attention, tokens
    
    def visualize_attention_heatmap(self, attention: np.ndarray, tokens: List[str], title: str = "Attention Patterns") -> go.Figure:
        """Create heatmap"""
        
        # Clean tokens
        display_tokens = [t.replace('Ġ', ' ').replace('Ċ', '↵') for t in tokens]
        
        fig = go.Figure(data=go.Heatmap(
            z=attention,
            x=display_tokens,
            y=display_tokens,
            colorscale='Reds',
            text=attention,
            texttemplate='%{z:.2f}',
            textfont={"size": 8}
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Key Tokens",
            yaxis_title="Query Tokens",
            height=600,
            template="plotly_white"
        )
        
        return fig
    
    def get_model_info(self) -> dict:
        """Get model architecture info"""
        config = self.model.config
        
        return {
            'num_layers': config.num_hidden_layers,
            'num_heads': config.num_attention_heads,
            'hidden_size': config.hidden_size,
            'vocab_size': config.vocab_size
        }
    
    