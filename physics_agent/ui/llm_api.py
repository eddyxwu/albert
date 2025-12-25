"""
LLM API interface for generating new gravitational theories

Primary support: xAI/Grok
Experimental support: OpenAI, Anthropic, Google Gemini
"""
import os
import requests
from typing import Optional

class LLMApi:
    """Handles LLM API calls for theory generation"""
    
    def __init__(self, provider: str = "grok"):
        self.provider = provider
        
        # API configuration based on provider
        if provider == "grok":
            self.api_key = os.getenv("GROK_API_KEY", "")
            self.base_url = "https://api.x.ai/v1"
            self.model = "grok-4"
        elif provider == "openai":
            # Experimental support
            self.api_key = os.getenv("OPENAI_API_KEY", "")
            self.base_url = "https://api.openai.com/v1"
            self.model = "gpt-4"
            print("Warning: OpenAI support is experimental. xAI/Grok is the primary supported provider.")
        elif provider == "anthropic":
            # Experimental support
            self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
            self.base_url = "https://api.anthropic.com/v1"
            self.model = "claude-4-opus"
            print("Warning: Anthropic support is experimental. xAI/Grok is the primary supported provider.")
        elif provider == "gemini":
            # Experimental support
            self.api_key = os.getenv("GOOGLE_API_KEY", "")
            self.base_url = "https://generativelanguage.googleapis.com/v1beta"
            self.model = "gemini-3-flash-preview"
            print("Warning: Google Gemini support is experimental. xAI/Grok is the primary supported provider.")
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Load prompt template - try simple one first for faster responses
        simple_template_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'self_discovery', 'simple_prompt_template.txt'
        )
        template_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'self_discovery', 'prompt_template.txt'
        )
        
        try:
            # Try simple template first
            if os.path.exists(simple_template_path):
                with open(simple_template_path, 'r') as f:
                    self.prompt_template = f.read()
            else:
                with open(template_path, 'r') as f:
                    self.prompt_template = f.read()
        except FileNotFoundError:
            # Fallback template
            self.prompt_template = """Generate a novel gravitational theory as a Python class inheriting from GravitationalTheory.

The new theory will be benchmarked against the following baseline theories:
{baseline_theories}

Your generated theory must:
- Be implemented as a Python class named 'CustomTheory' that inherits from 'GravitationalTheory'.
- Have a 'get_metric' method that returns the metric tensor components (g_tt, g_rr, g_pp, g_tp).
- Include a Lagrangian formulation of the theory in a docstring. This is a critical validation step.
- Aim to unify gravity and electromagnetism, or explore other novel geometric approaches to gravity.

Initial idea: {initial_prompt}
    
Return ONLY the Python code, no explanations."""
    
    def _fix_common_errors(self, code: str) -> str:
        """Fix common errors in generated code"""
        if not code:
            return code
            
        # Strip markdown code blocks
        if '```python' in code:
            # Extract code between ```python and ```
            start = code.find('```python') + 9
            end = code.find('```', start)
            if end > start:
                code = code[start:end].strip()
        elif '```' in code:
            # Strip any code blocks
            code = code.replace('```', '')
            
        # Fix incorrect imports
        code = code.replace('from gravitational_theory import', 'from physics_agent.base_theory import')
        code = code.replace('import gravitational_theory', 'import physics_agent.base_theory')
        
        # Ensure proper torch import
        if 'torch' in code and 'import torch' not in code:
            code = 'import torch\n' + code
            
        # Fix incorrect method signatures
        code = code.replace('get_metric(self, r, theta', 'get_metric(self, r')
        
        return code
    
    def generate_theory_variation(self, base_theory_code: str, modification_prompt: str, baseline_theories: list[str]) -> Optional[str]:
        """Generate a variation of a theory based on user input"""
        
        # Construct the prompt
        baseline_list = '\n'.join(f"{i+1}. {name}" for i, name in enumerate(baseline_theories))
        
        full_prompt = f"""Given this existing gravitational theory:

```python
{base_theory_code}
```

Please modify it according to this instruction: {modification_prompt}

The modified theory must:
- Be implemented as a Python class named 'CustomTheory' that inherits from 'GravitationalTheory'.
- Have a 'get_metric' method that returns the metric tensor components (g_tt, g_rr, g_pp, g_tp).
- Include a Lagrangian formulation of the theory in a docstring. This is a critical validation step.
- Maintain the core structure while applying the requested modification.

The theory will be benchmarked against:
{baseline_list}

Return ONLY the Python code, no explanations."""
        
        result = self._call_api(full_prompt)
        return self._fix_common_errors(result) if result else None
    
    def generate_new_theory(self, initial_prompt: str, baseline_theories: list[str]) -> Optional[str]:
        """Generate a completely new theory based on initial prompt"""
        
        baseline_list = '\n'.join(f"{i+1}. {name}" for i, name in enumerate(baseline_theories))
        prompt = self.prompt_template.format(
            baseline_theories=baseline_list,
            initial_prompt=initial_prompt if initial_prompt else 'Explore modifications to the Reissner-Nordström or Dilaton metric.'
        )
        
        result = self._call_api(prompt)
        return self._fix_common_errors(result) if result else None
    
    def refine_theory_with_feedback(
        self, 
        previous_code: str, 
        modification_prompt: str,
        validation_feedback: str,
        baseline_theories: list[str]
    ) -> Optional[str]:
        """
        Refine a theory based on validation feedback from Albert.
        
        This method is used in the Einstein Loop to iteratively improve theories
        that failed validation, feeding specific error messages back to the LLM.
        
        Args:
            previous_code: The code from the previous iteration
            modification_prompt: High-level instruction for refinement
            validation_feedback: Structured feedback from validators (failed tests, errors)
            baseline_theories: List of baseline theory names for comparison
            
        Returns:
            Refined theory code, or None if generation failed
        """
        baseline_list = '\n'.join(f"{i+1}. {name}" for i, name in enumerate(baseline_theories))
        
        full_prompt = f"""You are refining a gravitational theory that failed validation. 

## Previous Theory Code

```python
{previous_code}
```

## Validation Feedback

The theory was tested against Albert's physics validators and had these issues:

{validation_feedback}

## Your Task

{modification_prompt}

## Requirements

The refined theory must:
1. Be a Python class named 'CustomTheory' inheriting from 'GravitationalTheory'
2. Have a 'get_metric(self, r, M_param, C_param, G_param, **kwargs)' method returning (g_tt, g_rr, g_pp, g_tp)
3. Fix ALL the validation issues mentioned above
4. Maintain physical consistency (metric signature, asymptotic flatness, etc.)
5. Include a Lagrangian formulation in the class docstring

## Important Physics Constraints

- g_tt should be negative (timelike)
- g_rr should be positive (spacelike)  
- g_pp = r² for spherical symmetry
- At large r, metric should approach flat space: g_tt → -1, g_rr → 1
- Avoid singularities outside the event horizon

## Baseline Theories for Comparison

{baseline_list}

Return ONLY the corrected Python code. No explanations."""
        
        result = self._call_api(full_prompt)
        return self._fix_common_errors(result) if result else None
    
    def _call_api(self, prompt: str) -> Optional[str]:
        """Make the actual API call to the provider"""
        
        if not self.api_key:
            # Use correct env var name for each provider
            env_var_name = {
                "grok": "GROK_API_KEY",
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "gemini": "GOOGLE_API_KEY"
            }.get(self.provider, f"{self.provider.upper()}_API_KEY")
            print(f"Warning: {env_var_name} not set. Using mock response.")
            return self._mock_response()
        
        # Different API formats for different providers
        if self.provider == "grok":
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert theoretical physicist specializing in gravitational theories and general relativity."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "model": self.model,
                "stream": False,
                "temperature": 0.7
            }
            
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=300  # Increased timeout to 5 minutes
                )
                response.raise_for_status()
                
                result = response.json()
                return result['choices'][0]['message']['content']
                
            except Exception as e:
                print(f"API call failed: {e}")
                return None
                
        elif self.provider == "gemini":
            # Gemini API implementation
            headers = {
                "Content-Type": "application/json",
            }
            
            # Gemini uses a different format - system instruction in the prompt
            system_instruction = "You are an expert theoretical physicist specializing in gravitational theories and general relativity."
            full_prompt = f"{system_instruction}\n\n{prompt}"
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": full_prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 8192,
                    "topP": 0.95,
                    "topK": 40,
                }
            }
            
            # Try the primary model, with fallback to older models if needed
            models_to_try = [self.model]
            if self.model == "gemini-3.0-pro":
                # Add fallbacks if 3.0 isn't available
                models_to_try.extend(["gemini-1.5-pro", "gemini-pro"])
            elif self.model == "gemini-1.5-pro":
                models_to_try.append("gemini-pro")
            
            for model_name in models_to_try:
                try:
                    response = requests.post(
                        f"{self.base_url}/models/{model_name}:generateContent?key={self.api_key}",
                        headers=headers,
                        json=data,
                        timeout=300
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    
                    # Extract text from Gemini response
                    if 'candidates' in result and len(result['candidates']) > 0:
                        if 'content' in result['candidates'][0] and 'parts' in result['candidates'][0]['content']:
                            content = result['candidates'][0]['content']['parts'][0]['text']
                            if model_name != self.model:
                                print(f"Note: Using fallback model {model_name} (requested {self.model} not available)")
                            return content
                        else:
                            print(f"Gemini API: Unexpected response structure: {result.get('candidates', [{}])[0]}")
                            continue  # Try next model
                    elif 'error' in result:
                        error_msg = result['error'].get('message', str(result['error']))
                        # If it's a model not found error, try next model
                        if 'not found' in error_msg.lower() or '404' in str(result.get('error', {}).get('code', '')):
                            if model_name != models_to_try[-1]:  # Not the last fallback
                                print(f"Model {model_name} not available, trying fallback...")
                                continue
                        print(f"Gemini API error: {error_msg}")
                        return None
                    else:
                        print(f"Gemini API returned unexpected format: {result}")
                        continue  # Try next model
                        
                except requests.exceptions.RequestException as e:
                    if hasattr(e, 'response') and e.response is not None:
                        # If 404 (model not found), try next model
                        if e.response.status_code == 404 and model_name != models_to_try[-1]:
                            print(f"Model {model_name} not found, trying fallback...")
                            continue
                        try:
                            error_detail = e.response.json()
                            error_msg = error_detail.get('error', {}).get('message', str(error_detail))
                            if 'not found' in error_msg.lower() and model_name != models_to_try[-1]:
                                continue
                            print(f"Gemini API call failed: {error_msg}")
                        except:
                            print(f"Gemini API call failed: {e} (status: {e.response.status_code})")
                    else:
                        print(f"Gemini API call failed: {e}")
                    
                    # If this was the last model to try, return None
                    if model_name == models_to_try[-1]:
                        return None
                        
                except Exception as e:
                    print(f"Unexpected error calling Gemini API: {e}")
                    if model_name == models_to_try[-1]:
                        return None
                    continue
            
            return None
                
        elif self.provider in ["openai", "anthropic"]:
            # Experimental providers - not fully implemented
            print(f"Note: {self.provider} support is experimental. Full implementation pending.")
            print("Using mock response for now.")
            return self._mock_response()
        else:
            print(f"Unsupported provider: {self.provider}")
            return None
    
    def _mock_response(self) -> str:
        """Return a mock theory for testing when API key is not available"""
        return '''import torch
import sympy as sp
from physics_agent.base_theory import GravitationalTheory

class CustomTheory(GravitationalTheory):
    """
    Modified Schwarzschild metric with small torsion-inspired correction
    
    Physical Motivation:
    This theory explores a simple modification to Schwarzschild geometry
    inspired by Einstein-Cartan theory with torsion. The correction term
    represents the leading-order effect of spacetime torsion on the metric.
    
    Lagrangian: L = R + epsilon * T^2
    where R is the Ricci scalar and T represents torsion contributions.
    The correction is small (epsilon << 1) to maintain compatibility with
    Solar System tests.
    
    Expected Effects:
    - Small corrections to perihelion precession
    - Modified light deflection at higher orders
    - Stable circular orbits maintained
    """
    
    def __init__(self, epsilon: float = 1e-6):
        super().__init__(name=f"Torsion Modified (ε={epsilon})", force_6dof_solver=False)
        self.epsilon = epsilon
        self.category = "classical"
        
        # Define symbolic Lagrangian
        R = sp.Symbol('R')
        T = sp.Symbol('T')  # Torsion scalar
        self.lagrangian = R + self.epsilon * T**2
    
    def get_metric(self, r, M_param, C_param, G_param, **kwargs):
        """
        Compute metric components with torsion correction.
        
        The correction is designed to:
        1. Vanish at large r (asymptotic flatness)
        2. Be small enough to pass Solar System tests
        3. Maintain proper metric signature
        """
        # Schwarzschild radius
        rs = 2 * G_param * M_param / C_param**2
        
        # Base Schwarzschild metric function
        f_base = 1 - rs / r
        
        # Torsion correction: decays as 1/r^3 for asymptotic flatness
        # Scaled by epsilon to keep correction small
        torsion_correction = self.epsilon * (rs / r)**3
        
        # Modified metric function
        f = f_base * (1 + torsion_correction)
        
        # Ensure f stays positive outside horizon (add small floor)
        f = torch.clamp(f, min=1e-10)
        
        # Metric components
        g_tt = -f  # Negative for timelike
        g_rr = 1 / f  # Positive for spacelike
        g_pp = r**2  # Standard angular part
        g_tp = torch.zeros_like(r)  # Static spacetime
        
        return g_tt, g_rr, g_pp, g_tp
''' 