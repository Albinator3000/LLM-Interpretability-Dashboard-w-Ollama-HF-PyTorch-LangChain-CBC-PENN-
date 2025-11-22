from langchain_community.llms import Ollama
from langchain.agents import AgentExecutor, create_react_agent
from lanchain.tools import Tool
from langchain.callbacks.base import BaseCallbackHandler
from langchain import hub
from typing import List, Dict, Any
import time

class TracingCallback(BaseCallbackHandler):
    """Track agent decisions"""
    
    def __init__(self):
        self.steps = []
        self.current_step = {}
    
    def on_agent_action(self, action, **kwargs):
        """Called when agent acts"""
        self.current_step = {
            'type': 'action',
            'tool': action.tool,
            'tool_input': action.tool_input,
            'log': action.log,
            'timestamp': time.time()
        }
        self.steps.append(self.current_step.copy())
    
    def on_tool_end(self, output, **kwargs):
        """Called when tool finishes"""
        self.steps.append({
            'type': 'observation',
            'output': str(output),
            'timestamp': time.time()
        })
    
    def on_agent_finish(self, finish, **kwargs):
        """Called when agent finishes"""
        self.steps.append({
            'type': 'finish',
            'output': finish.return_values,
            'timestamp': time.time()
        })
    
    def clear(self):
        """Reset trace"""
        self.steps = []


class AgentTracer:
    """Trace agent reasoning"""
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.model_name = model_name
        self.llm = Ollama(model=model_name, temperature=0.7)
        self.tools = self._create_tools()
        self.callback = TracingCallback()
    
    def _create_tools(self) -> List[Tool]:
        """Create tools"""
        
        def calculator(expression: str) -> str:
            try:
                result = eval(expression, {"__builtins__": {}}, {})
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {str(e)}"
        
        def word_counter(text: str) -> str:
            count = len(text.split())
            return f"Word count: {count}"
        
        def reverse_text(text: str) -> str:
            return text[::-1]
        
        def length_calculator(text: str) -> str:
            return f"Length: {len(text)} characters"
        
        return [
            Tool(
                name="Calculator",
                func=calculator,
                description="Useful for math. Input: valid Python expression like '5 * 3 + 2'"
            ),
            Tool(
                name="WordCounter",
                func=word_counter,
                description="Counts words. Input: text to count"
            ),
            Tool(
                name="ReverseText",
                func=reverse_text,
                description="Reverses text. Input: text to reverse"
            ),
            Tool(
                name="LengthCalculator",
                func=length_calculator,
                description="Calculates length. Input: text"
            )
        ]
    
    def run_agent(self, task: str) -> Dict[str, Any]:
        """Run agent and return trace"""
        
        self.callback.clear()
        
        # Get prompt
        prompt = hub.pull("hwchase17/react")
        
        # Create agent
        agent = create_react_agent(self.llm, self.tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
        
        # Run
        try:
            result = agent_executor.invoke(
                {"input": task},
                {"callbacks": [self.callback]}
            )
            
            return {
                'success': True,
                'output': result.get('output', 'No output'),
                'steps': self.callback.steps
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'steps': self.callback.steps
            }
    
    def format_trace_for_display(self, steps: List[Dict]) -> List[Dict]:
        """Format for Streamlit"""
        formatted = []
        
        for i, step in enumerate(steps):
            if step['type'] == 'action':
                formatted.append({
                    'step_num': i + 1,
                    'type': '🤔 Reasoning',
                    'content': step['log'],
                    'tool': step['tool'],
                    'input': step['tool_input']
                })
            elif step['type'] == 'observation':
                formatted.append({
                    'step_num': i + 1,
                    'type': '👁️ Observation',
                    'content': step['output']
                })
            elif step['type'] == 'finish':
                formatted.append({
                    'step_num': i + 1,
                    'type': '✅ Final Answer',
                    'content': step['output']
                })
        
        return formatted