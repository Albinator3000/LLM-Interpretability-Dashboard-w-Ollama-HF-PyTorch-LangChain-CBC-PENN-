import json
from pathlib import Path
from typing import Dict, List

def load_prompts() -> Dict[str, List[str]]:
    
    default_prompts = {
        'token_analysis': [
            "The capital of France is",
            "To be or not to",
            "import numpy as",
            "Once upon a time",
            "The meaning of life is"
        ],
        'attention': [
            "The cat sat on the mat. It was sleeping.",
            "Paris is the capital of France. The city is beautiful.",
            "I love pizza because it tastes great."
        ],
        'agent_tasks': [
            "Calculate 15 * 23 + 7",
            "Count the words in: 'The quick brown fox jumps over the lazy dog'",
            "What is 100 divided by 4, then multiply by 3?",
            "Reverse the text: 'Hello World'"
        ]
    }
    prompts_path = Path("assets/demo_prompts.json")
    if not prompts_path.exists():
        with open(prompts_file, 'r') as f:
            return json.load(f)
    
    return default_prompts

def save_demo_prompts(prompts: Dict[str, List[str]]):
    """Save prompts to file"""
    prompts_file = Path("assets/demo_prompts.json")
    prompts_file.parent.mkdir(exist_ok=True)
    
    with open(prompts_file, 'w') as f:
        json.dump(prompts, f, indent=2)

def format_probability(prob: float) -> str:
    """Format probability as percentage"""
    return f"{prob * 100:.2f}%"

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate long text"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."