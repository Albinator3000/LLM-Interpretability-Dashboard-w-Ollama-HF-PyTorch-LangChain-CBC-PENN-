"""
Agent Tracer - Track LLM agent reasoning with Ollama
"""

import ollama
from typing import List, Dict, Any
import time


class AgentTracer:
    """Simple agent tracer using Ollama directly"""

    def __init__(self, model_name: str = "llama3.2:3b"):
        self.model_name = model_name
        self.tools = self._create_tools()

    def _create_tools(self) -> Dict[str, callable]:
        """Create available tools"""

        def calculator(expression: str) -> str:
            try:
                # Safe eval for basic math
                allowed = {
                    '__builtins__': {},
                    'abs': abs, 'round': round,
                    'min': min, 'max': max,
                    'sum': sum, 'pow': pow
                }
                result = eval(expression, allowed, {})
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {str(e)}"

        def word_counter(text: str) -> str:
            count = len(text.split())
            return f"Word count: {count}"

        def reverse_text(text: str) -> str:
            return f"Reversed: {text[::-1]}"

        def length_calculator(text: str) -> str:
            return f"Length: {len(text)} characters"

        return {
            "calculator": calculator,
            "word_counter": word_counter,
            "reverse_text": reverse_text,
            "length_calculator": length_calculator
        }

    def run_agent(self, task: str) -> Dict[str, Any]:
        """Run agent with simple ReAct-style reasoning"""

        steps = []

        system_prompt = """You are a helpful assistant that solves tasks step by step.

Available tools:
- calculator: Evaluate math expressions (e.g., "15 * 23 + 7")
- word_counter: Count words in text
- reverse_text: Reverse a string
- length_calculator: Get character count

For each step, respond in this format:
Thought: [your reasoning]
Action: [tool_name]
Input: [input for the tool]

When you have the final answer:
Thought: [final reasoning]
Final Answer: [your answer]

Be concise."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task}
        ]

        try:
            max_iterations = 5

            for i in range(max_iterations):
                # Get model response
                response = ollama.chat(
                    model=self.model_name,
                    messages=messages
                )

                content = response['message']['content']
                steps.append({
                    'type': 'reasoning',
                    'content': content,
                    'timestamp': time.time()
                })

                # Check for final answer
                if "Final Answer:" in content:
                    final_answer = content.split("Final Answer:")[-1].strip()
                    return {
                        'success': True,
                        'output': final_answer,
                        'steps': steps
                    }

                # Parse action and input
                if "Action:" in content and "Input:" in content:
                    try:
                        action_line = [l for l in content.split('\n') if 'Action:' in l][0]
                        input_line = [l for l in content.split('\n') if 'Input:' in l][0]

                        action = action_line.split('Action:')[-1].strip().lower()
                        tool_input = input_line.split('Input:')[-1].strip()

                        # Execute tool
                        if action in self.tools:
                            result = self.tools[action](tool_input)
                            steps.append({
                                'type': 'observation',
                                'tool': action,
                                'input': tool_input,
                                'output': result,
                                'timestamp': time.time()
                            })

                            # Add observation to messages
                            messages.append({"role": "assistant", "content": content})
                            messages.append({"role": "user", "content": f"Observation: {result}"})
                        else:
                            messages.append({"role": "assistant", "content": content})
                            messages.append({"role": "user", "content": f"Error: Unknown tool '{action}'. Available: calculator, word_counter, reverse_text, length_calculator"})
                    except Exception as e:
                        messages.append({"role": "assistant", "content": content})
                        messages.append({"role": "user", "content": f"Error parsing action: {e}. Please use the correct format."})
                else:
                    # No action found, ask to continue
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": "Please continue with an Action or provide your Final Answer."})

            return {
                'success': True,
                'output': "Max iterations reached. Last response: " + content,
                'steps': steps
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'steps': steps
            }

    def format_trace_for_display(self, steps: List[Dict]) -> List[Dict]:
        """Format steps for Streamlit display"""
        formatted = []

        for i, step in enumerate(steps):
            if step['type'] == 'reasoning':
                formatted.append({
                    'step_num': i + 1,
                    'type': 'Reasoning',
                    'content': step['content'],
                    'tool': '',
                    'input': ''
                })
            elif step['type'] == 'observation':
                formatted.append({
                    'step_num': i + 1,
                    'type': 'Observation',
                    'content': f"Tool: {step['tool']}\nInput: {step['input']}\nOutput: {step['output']}"
                })

        return formatted
