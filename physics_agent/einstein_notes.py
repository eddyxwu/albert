#!/usr/bin/env python3
"""
Einstein Notes Generator

Generates candidate completions of Einstein's bedside notes that pass
all of Albert's physics validators using an LLM feedback loop.

Usage:
    python -m physics_agent.einstein_notes --prompt "unified field theory"
    python -m physics_agent.einstein_notes --count 3 --threshold 0.90
"""

import os
import sys
import json
import argparse
import importlib.util
from datetime import datetime
from typing import Optional, Tuple, Dict, List

from physics_agent.evaluation import test_theory_comprehensive
from physics_agent.ui.llm_api import LLMApi
from physics_agent.base_theory import GravitationalTheory


CANDIDATES_DIR = os.path.join(os.path.dirname(__file__), 'theories', 'candidates', 'einstein_notes')
BASELINES = ["Schwarzschild", "Kerr", "Reissner-Nordstrom"]

SIGNATURE_HINT = """IMPORTANT: get_metric must have this EXACT signature:
def get_metric(self, r, M_param, C_param, G_param, **kwargs):
    return g_tt, g_rr, g_pp, g_tp"""

# Fallback starter code when first generation fails - has a simple modification
FALLBACK_STARTER = '''import torch
from physics_agent.base_theory import GravitationalTheory

class CustomTheory(GravitationalTheory):
    """Modified metric with power-law correction as starting point."""
    category = "modified_gravity"
    
    def __init__(self, alpha=0.01):
        super().__init__(name="CustomTheory", force_6dof_solver=False)
        self.alpha = alpha  # NON-ZERO correction strength
    
    def get_metric(self, r, M_param, C_param, G_param, **kwargs):
        rs = 2 * G_param * M_param / C_param**2
        # Power-law correction term (not pure Schwarzschild!)
        correction = self.alpha * (rs/r)**2
        f = 1 - rs/r + correction
        f = torch.maximum(f, torch.tensor(1e-10, device=r.device))
        g_tt = -f
        g_rr = 1/f
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        return g_tt, g_rr, g_pp, g_tp
'''


CREATIVE_APPROACHES = [
    "exponential: alpha * torch.exp(-r/lambda_scale)",
    "power-law: alpha * (rs/r)**n where n = 2, 3, or 0.5",
    "logarithmic: alpha * torch.log(r/rs) / r",
    "Yukawa: alpha * torch.exp(-r/lambda_scale) / r",
    "polynomial: alpha * (rs/r) + beta * (rs/r)**2",
    "tanh transition: alpha * torch.tanh((r - r0) / delta)",
    "inverse-square: alpha / (r**2 + epsilon)",
    "oscillatory: alpha * torch.sin(r/lambda_scale) / r**2",
]

def build_refinement_context(best_code: str, best_rate: float, errors: str, iteration: int = 1) -> str:
    """Build rich context for LLM refinement with creativity hints."""
    # Rotate through creative suggestions based on iteration
    approach_hint = CREATIVE_APPROACHES[(iteration - 1) % len(CREATIVE_APPROACHES)]
    
    return f"""Your best working theory so far ({best_rate*100:.1f}% pass rate):

```python
{best_code}
```

{errors}

**TRY A DIFFERENT MATHEMATICAL APPROACH!**
Don't just tweak parameters - try a completely different correction formula.
Suggested approach for this iteration: {approach_hint}

{SIGNATURE_HINT}

Generate a theory with a DIFFERENT mathematical form. Return ONLY Python code."""


def load_theory_from_code(code: str, session_dir: str, iteration: int) -> Optional[type]:
    """Load a GravitationalTheory subclass from code string."""
    os.makedirs(session_dir, exist_ok=True)
    theory_path = os.path.join(session_dir, f'theory_v{iteration}.py')
    
    with open(theory_path, 'w') as f:
        f.write(code)
    
    try:
        spec = importlib.util.spec_from_file_location(f"theory_v{iteration}_{id(code)}", theory_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, GravitationalTheory) and obj != GravitationalTheory:
                return obj
        return None
    except Exception as e:
        print(f"  Load error: {e}")
        return None


def extract_error_summary(results: Dict) -> str:
    """Extract failed tests into LLM-readable format."""
    errors = []
    
    for test in results.get('analytical_tests', []) + results.get('solver_tests', []):
        if not test.get('passed', False) and test.get('status') not in ['SKIP', 'N/A']:
            msg = f"- {test['name']}: FAILED"
            if test.get('predicted_value') is not None and test.get('observed_value') is not None:
                msg += f" (predicted {test['predicted_value']:.4g}, expected {test['observed_value']:.4g})"
            elif test.get('error_percent') is not None:
                msg += f" (error: {test['error_percent']:.1f}%)"
            elif test.get('notes'):
                msg += f" ({test['notes'][:100]})"
            errors.append(msg)
    
    if not errors:
        return "All tests passed!"
    return "Fix these validation failures:\n" + "\n".join(errors)


def generate_candidate(
    llm_api: LLMApi,
    prompt: str,
    session_id: str,
    max_iterations: int = 10,
    threshold: float = 0.85
) -> Tuple[Optional[str], Dict, int]:
    """Generate a single candidate that passes validation."""
    session_dir = os.path.join(CANDIDATES_DIR, session_id)
    
    print(f"\n{'='*60}")
    print(f"Session: {session_id}")
    print(f"Prompt: {prompt[:60]}...")
    print(f"{'='*60}")
    
    # Initial generation
    print("\nIteration 1: Generating initial theory...")
    code = llm_api.generate_new_theory(prompt, BASELINES)
    
    if not code:
        print("ERROR: LLM failed to generate code")
        return None, {}, 0
    
    # Track best candidate and last working code
    best_code = None
    best_rate = 0.0
    best_results = {}
    last_working_code = None
    last_errors = ""
    
    results = {}
    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration}/{max_iterations} ---")
        
        # Load theory
        theory_class = load_theory_from_code(code, session_dir, iteration)
        if theory_class is None:
            # Use fallback starter if no working code exists yet
            base_code = last_working_code or FALLBACK_STARTER
            print(f"  Syntax error, using {'last working' if last_working_code else 'fallback starter'} code...")
            refinement = build_refinement_context(base_code, best_rate, f"Syntax error in generated code. Start fresh from this working example.\n{last_errors}", iteration)
            code = llm_api.generate_theory_variation(base_code, refinement, BASELINES)
            if not code:
                break
            continue
        
        last_working_code = code  # Save working code
        
        # Run ALL validators
        try:
            theory = theory_class()
            category = getattr(theory_class, 'category', 'classical')
            results = test_theory_comprehensive(theory.name, theory_class, category)
        except Exception as e:
            print(f"  Validation error: {e}")
            refinement = build_refinement_context(code, best_rate, f"Runtime error: {e}\n{SIGNATURE_HINT}", iteration)
            code = llm_api.generate_theory_variation(code, refinement, BASELINES)
            if not code:
                break
            continue
        
        if results is None:
            print("  Theory initialization failed")
            refinement = build_refinement_context(code, best_rate, "Theory failed to initialize", iteration)
            code = llm_api.generate_theory_variation(code, refinement, BASELINES)
            if not code:
                break
            continue
        
        pass_rate = results['combined_summary']['success_rate']
        print(f"\n  Pass rate: {pass_rate*100:.1f}%")
        
        # Track best
        if pass_rate > best_rate:
            best_code = code
            best_rate = pass_rate
            best_results = results
            print(f"  New best! ({best_rate*100:.1f}%)")
        
        if pass_rate >= threshold:
            print(f"\n  SUCCESS! Passed {pass_rate*100:.1f}% of validators")
            with open(os.path.join(session_dir, 'validation.json'), 'w') as f:
                json.dump({'pass_rate': pass_rate, 'iteration': iteration}, f, indent=2)
            return code, results, iteration
        
        # Build rich context for refinement
        last_errors = extract_error_summary(results)
        refinement = build_refinement_context(best_code or code, best_rate, last_errors, iteration)
        print(f"\n  Refining with new approach...")
        code = llm_api.generate_theory_variation(best_code or code, refinement, BASELINES)
        if not code:
            print("  ERROR: LLM failed to generate refinement")
            break
    
    # Return best candidate even if below threshold
    print(f"\n  Returning best candidate ({best_rate*100:.1f}%)")
    return best_code, best_results, max_iterations


def main():
    parser = argparse.ArgumentParser(description="Generate Einstein's bedside notes completions")
    parser.add_argument('--prompt', type=str, default="Complete Einstein's unified field theory notes",
                        help='Initial prompt for theory generation')
    parser.add_argument('--count', type=int, default=1, help='Number of candidates to generate')
    parser.add_argument('--threshold', type=float, default=0.85, help='Minimum pass rate (0-1)')
    parser.add_argument('--max-iterations', type=int, default=10, help='Max refinement iterations')
    parser.add_argument('--provider', type=str, default='grok', choices=['grok', 'openai', 'anthropic', 'gemini'],
                        help='LLM provider')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("EINSTEIN NOTES GENERATOR")
    print("="*70)
    print(f"Candidates: {args.count} | Threshold: {args.threshold*100:.0f}% | Provider: {args.provider}")
    
    llm_api = LLMApi(provider=args.provider)
    session_base = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    successful = []
    for i in range(args.count):
        session_id = f"{session_base}_candidate_{i+1}"
        code, results, iterations = generate_candidate(
            llm_api, args.prompt, session_id, args.max_iterations, args.threshold
        )
        if code and results.get('combined_summary', {}).get('success_rate', 0) >= args.threshold:
            successful.append({'session_id': session_id, 'pass_rate': results['combined_summary']['success_rate']})
    
    print("\n" + "="*70)
    print(f"RESULTS: {len(successful)}/{args.count} successful candidates")
    for s in successful:
        print(f"  - {s['session_id']}: {s['pass_rate']*100:.1f}%")
    
    return successful


if __name__ == "__main__":
    main()

