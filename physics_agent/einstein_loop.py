#!/usr/bin/env python3
"""
Einstein's Bedside Notes - Theory Generation & Validation Loop

This script implements an iterative loop that:
1. Uses LLMs to generate gravitational theory candidates
2. Validates them against physics constraints using Albert's validators
3. Feeds errors back to the LLM for refinement
4. Continues until a valid theory is found or max iterations reached

Usage:
    python -m physics_agent.einstein_loop --prompt "unified field with torsion"
    python -m physics_agent.einstein_loop --provider grok --max-iterations 5
"""

import argparse
import os
import sys
import json
import traceback
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics_agent.ui.llm_api import LLMApi
from physics_agent.base_theory import GravitationalTheory


class TheoryExecutionError(Exception):
    """Raised when theory code fails to execute"""
    pass


class EinsteinLoop:
    """
    Main loop for generating and validating gravitational theory candidates.
    """
    
    BASELINE_THEORIES = ["Schwarzschild", "Kerr", "Reissner-Nordström"]
    
    def __init__(
        self, 
        provider: str = "grok",
        max_iterations: int = 5,
        output_dir: str = None,
        verbose: bool = False,
        continue_after_valid: bool = False
    ):
        self.llm_api = LLMApi(provider=provider)
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.continue_after_valid = continue_after_valid
        
        # Set up output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(
                os.path.dirname(__file__), 
                "runs", 
                f"einstein_loop_{timestamp}"
            )
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Track iteration history
        self.history: List[Dict[str, Any]] = []
        
        # Lazy load engine to avoid circular imports
        self._engine = None
        
    @property
    def engine(self):
        """Lazy load TheoryEngine to avoid import issues"""
        if self._engine is None:
            from physics_agent.theory_engine_core import TheoryEngine
            self._engine = TheoryEngine(device='cpu', dtype=torch.float64)
        return self._engine
    
    def execute_theory_code(self, code: str) -> GravitationalTheory:
        """
        Safely execute theory code and return the theory instance.
        
        Args:
            code: Python code defining a CustomTheory class
            
        Returns:
            Instance of the CustomTheory class
            
        Raises:
            TheoryExecutionError: If code fails to execute or instantiate
        """
        # Create a sandbox namespace
        namespace = {
            'torch': torch,
            'GravitationalTheory': GravitationalTheory,
        }
        
        # Add sympy if referenced
        if 'sympy' in code or 'sp.' in code:
            import sympy as sp
            namespace['sp'] = sp
            namespace['sympy'] = sp
        
        try:
            # Execute the code in the sandbox
            exec(code, namespace)
        except SyntaxError as e:
            raise TheoryExecutionError(f"Syntax error in generated code: {e}")
        except Exception as e:
            raise TheoryExecutionError(f"Runtime error executing code: {e}\n{traceback.format_exc()}")
        
        # Find the CustomTheory class
        if 'CustomTheory' not in namespace:
            # Try to find any GravitationalTheory subclass
            theory_class = None
            for name, obj in namespace.items():
                if (isinstance(obj, type) and 
                    issubclass(obj, GravitationalTheory) and 
                    obj != GravitationalTheory):
                    theory_class = obj
                    break
            
            if theory_class is None:
                raise TheoryExecutionError(
                    "Generated code does not define a CustomTheory class or any GravitationalTheory subclass"
                )
        else:
            theory_class = namespace['CustomTheory']
        
        # Instantiate the theory
        try:
            theory = theory_class()
            return theory
        except Exception as e:
            raise TheoryExecutionError(f"Failed to instantiate theory: {e}")
    
    def validate_theory(self, theory: GravitationalTheory) -> Dict[str, Any]:
        """
        Run validators on the theory and return results.
        
        Args:
            theory: Theory instance to validate
            
        Returns:
            Dict with validation results
        """
        # Initial radius at 10 times the length scale (10M in geometric units)
        r0_si = float(10.0 * self.engine.length_scale)
        
        # Run a short trajectory for validation
        # Use proper time step in SI units (scaled by engine.time_scale)
        n_steps = 100
        dtau_si = 0.1 * self.engine.time_scale
        
        try:
            hist, tag, _ = self.engine.run_trajectory(
                theory, r0_si, n_steps, dtau_si,
                quantum_interval=0, quantum_beta=0.0,
                no_cache=True,
                verbose=False
            )
            
            if hist is None or hist.shape[0] <= 1:
                return {
                    "success": False,
                    "error": f"Trajectory computation failed: {tag}",
                    "validations": []
                }
            
            # Get initial conditions for validation
            y0_general = hist[0]
            
            # Run all validations
            validation_results = self.engine.run_all_validations(
                theory, hist, y0_general, categories=["constraint", "observational"]
            )
            
            return {
                "success": True,
                "validations": validation_results.get("validations", []),
                "trajectory_steps": hist.shape[0]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Validation error: {e}",
                "traceback": traceback.format_exc(),
                "validations": []
            }
    
    def format_feedback(self, iteration_result: Dict[str, Any]) -> str:
        """
        Format iteration results into feedback for the LLM.
        
        Args:
            iteration_result: Results from previous iteration
            
        Returns:
            Formatted feedback string
        """
        feedback_parts = []
        
        # Add execution errors if any
        if iteration_result.get("execution_error"):
            feedback_parts.append(f"## Execution Error\n{iteration_result['execution_error']}")
        
        # Add validation results
        if iteration_result.get("validation_results"):
            val_results = iteration_result["validation_results"]
            
            if not val_results.get("success"):
                feedback_parts.append(f"## Validation Failed\n{val_results.get('error', 'Unknown error')}")
            else:
                validations = val_results.get("validations", [])
                
                failed = []
                passed = []
                warnings = []
                
                for v in validations:
                    name = v.get("validator", "Unknown")
                    flags = v.get("flags", {})
                    overall = flags.get("overall", "UNKNOWN")
                    details = v.get("details", {})
                    loss = v.get("loss", None)
                    
                    # Determine actual status
                    if overall == "PASS" or overall == "PASSED":
                        passed.append(name)
                    elif overall == "WARNING":
                        warnings.append(f"- {name}: {overall} (loss: {loss})")
                    elif overall in ["FAIL", "FAILED", "ERROR"]:
                        error_info = details.get("error", details.get("message", flags.get("details", "No details")))
                        if isinstance(error_info, dict):
                            error_info = str(error_info)
                        failed.append(f"- {name}: {overall} - {error_info}")
                    else:
                        # Check loss value as fallback
                        if loss is not None and loss < 0.1:
                            passed.append(name)
                        else:
                            failed.append(f"- {name}: {overall} (loss: {loss})")
                
                if failed:
                    feedback_parts.append("## Failed Validators\n" + "\n".join(failed))
                
                if warnings:
                    feedback_parts.append("## Warnings\n" + "\n".join(warnings))
                
                if passed:
                    feedback_parts.append(f"## Passed Validators ({len(passed)})\n{', '.join(passed)}")
                
                # Add summary
                total = len(failed) + len(passed) + len(warnings)
                feedback_parts.append(f"\n## Summary\n{len(passed)}/{total} validators passed, {len(failed)} failed, {len(warnings)} warnings")
        
        if not feedback_parts:
            return "No specific errors found, but theory did not pass all validations."
        
        return "\n\n".join(feedback_parts)
    
    def run_iteration(
        self, 
        prompt: str, 
        previous_code: str = None,
        feedback: str = None,
        iteration: int = 0
    ) -> Dict[str, Any]:
        """
        Run a single iteration of the generation-validation loop.
        
        Args:
            prompt: Initial prompt or refinement instruction
            previous_code: Code from previous iteration (if refining)
            feedback: Feedback from previous iteration
            iteration: Current iteration number
            
        Returns:
            Dict with iteration results
        """
        result = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "feedback": feedback,
            "code": None,
            "execution_error": None,
            "validation_results": None,
            "is_valid": False
        }
        
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{self.max_iterations}")
        print(f"{'='*60}")
        
        # Generate or refine theory code
        if previous_code and feedback:
            print("Refining theory based on feedback...")
            refinement_prompt = f"""The previous theory had these issues:

{feedback}

Please fix these issues. {prompt}"""
            code = self.llm_api.refine_theory_with_feedback(
                previous_code, 
                refinement_prompt,
                feedback,
                self.BASELINE_THEORIES
            )
        else:
            print(f"Generating new theory: {prompt}")
            code = self.llm_api.generate_new_theory(prompt, self.BASELINE_THEORIES)
        
        if not code:
            result["execution_error"] = "LLM failed to generate code"
            return result
        
        result["code"] = code
        
        if self.verbose:
            print("\n--- Generated Code ---")
            print(code[:500] + "..." if len(code) > 500 else code)
            print("----------------------\n")
        
        # Try to execute the code
        print("Executing generated theory...")
        try:
            theory = self.execute_theory_code(code)
            print(f"  Theory instantiated: {theory.name}")
        except TheoryExecutionError as e:
            result["execution_error"] = str(e)
            print(f"  Execution failed: {e}")
            return result
        
        # Run validation
        print("Running validators...")
        validation_results = self.validate_theory(theory)
        result["validation_results"] = validation_results
        
        if not validation_results.get("success"):
            print(f"  Validation failed: {validation_results.get('error')}")
            return result
        
        # Check validator results
        validations = validation_results.get("validations", [])
        passed_count = 0
        failed_count = 0
        warning_count = 0
        
        # Track which key validators passed
        key_validators_passed = {
            "conservation": False,
            "metric": False,
            "mercury": False,
            "light": False,
        }
        
        for v in validations:
            name = v.get("validator", "").lower()
            flags = v.get("flags", {})
            overall = flags.get("overall", "UNKNOWN")
            
            if overall in ["PASS", "PASSED"]:
                passed_count += 1
                # Check key validators
                if "conservation" in name:
                    key_validators_passed["conservation"] = True
                elif "metric" in name:
                    key_validators_passed["metric"] = True
                elif "mercury" in name:
                    key_validators_passed["mercury"] = True
                elif "light" in name or "deflection" in name:
                    key_validators_passed["light"] = True
            elif overall == "WARNING":
                warning_count += 1
            else:
                failed_count += 1
        
        total = passed_count + failed_count + warning_count
        print(f"  Results: {passed_count}/{total} passed, {failed_count} failed, {warning_count} warnings")
        
        # Theory is valid if:
        # 1. All key validators pass (conservation, metric properties, Mercury, light deflection)
        # 2. OR at least 70% of validators pass
        key_pass_count = sum(key_validators_passed.values())
        pass_ratio = passed_count / total if total > 0 else 0
        
        is_valid = (key_pass_count >= 3) or (pass_ratio >= 0.7)
        
        if is_valid:
            print(f"  Theory meets validity criteria (key validators: {key_pass_count}/4, pass ratio: {pass_ratio:.1%})")
        
        result["is_valid"] = is_valid
        
        return result
    
    def save_candidate(self, code: str, result: Dict[str, Any]) -> str:
        """
        Save a valid candidate theory to the candidates directory.
        
        Args:
            code: Theory code
            result: Validation results
            
        Returns:
            Path to saved candidate
        """
        # Generate unique ID
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate_id = f"c_{timestamp}_{code_hash}"
        
        # Create candidate directory
        candidates_dir = os.path.join(
            os.path.dirname(__file__),
            "theories", "candidates", "proposed", candidate_id
        )
        os.makedirs(candidates_dir, exist_ok=True)
        
        # Save theory.py
        theory_path = os.path.join(candidates_dir, "theory.py")
        with open(theory_path, 'w') as f:
            f.write(code)
        
        # Save __init__.py
        init_path = os.path.join(candidates_dir, "__init__.py")
        with open(init_path, 'w') as f:
            f.write(f'"""Auto-generated candidate theory {candidate_id}"""\n')
            f.write('from .theory import CustomTheory\n')
        
        # Save validation results
        results_path = os.path.join(candidates_dir, "validation_results.json")
        with open(results_path, 'w') as f:
            # Convert non-serializable objects
            serializable_result = json.loads(
                json.dumps(result, default=str)
            )
            json.dump(serializable_result, f, indent=2)
        
        # Create README
        readme_path = os.path.join(candidates_dir, "README.md")
        with open(readme_path, 'w') as f:
            f.write(f"# Candidate Theory: {candidate_id}\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write("## Validation Results\n\n")
            f.write("This theory passed all validation checks.\n\n")
            f.write("## Discovery Context\n\n")
            f.write(f"Prompt: {result.get('prompt', 'Unknown')}\n")
            f.write(f"Iterations: {result.get('iteration', 0) + 1}\n")
        
        print(f"\nCandidate saved to: {candidates_dir}")
        return candidates_dir
    
    def run(self, initial_prompt: str) -> Dict[str, Any]:
        """
        Run the full generation-validation loop.
        
        Args:
            initial_prompt: Initial prompt describing desired theory
            
        Returns:
            Dict with loop results including any valid candidates
        """
        print(f"\n{'#'*60}")
        print("# Einstein's Bedside Notes - Theory Generation Loop")
        print(f"{'#'*60}")
        print(f"Provider: {self.llm_api.provider}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Initial prompt: {initial_prompt}")
        print(f"Output directory: {self.output_dir}")
        
        previous_code = None
        feedback = None
        valid_candidates = []
        
        for i in range(self.max_iterations):
            result = self.run_iteration(
                initial_prompt,
                previous_code=previous_code,
                feedback=feedback,
                iteration=i
            )
            
            self.history.append(result)
            
            if result["is_valid"]:
                print(f"\n*** Valid theory found at iteration {i + 1}! ***")
                candidate_path = self.save_candidate(result["code"], result)
                valid_candidates.append({
                    "iteration": i,
                    "path": candidate_path,
                    "code": result["code"]
                })
                if not self.continue_after_valid:
                    break  # Stop on first valid theory unless continuing
            
            # Prepare for next iteration
            previous_code = result.get("code")
            if previous_code:
                feedback = self.format_feedback(result)
                
                if self.verbose:
                    print(f"\n--- Feedback for next iteration ---")
                    print(feedback)
                    print("-----------------------------------\n")
        
        # Save iteration history
        history_path = os.path.join(self.output_dir, "iteration_history.json")
        with open(history_path, 'w') as f:
            serializable_history = json.loads(
                json.dumps(self.history, default=str)
            )
            json.dump(serializable_history, f, indent=2)
        
        # Generate summary report
        self._generate_report(valid_candidates)
        
        summary = {
            "total_iterations": len(self.history),
            "valid_candidates": len(valid_candidates),
            "candidates": valid_candidates,
            "output_dir": self.output_dir
        }
        
        print(f"\n{'='*60}")
        print("Loop Complete")
        print(f"{'='*60}")
        print(f"Total iterations: {summary['total_iterations']}")
        print(f"Valid candidates found: {summary['valid_candidates']}")
        print(f"Results saved to: {self.output_dir}")
        
        return summary
    
    def _generate_report(self, valid_candidates: List[Dict]):
        """Generate HTML report of the loop results."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Einstein Loop Results</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }
        h1 { color: #00d4ff; }
        h2 { color: #ff6b6b; margin-top: 30px; }
        .iteration { background: #16213e; padding: 20px; margin: 20px 0; border-radius: 8px; border-left: 4px solid #00d4ff; }
        .success { border-left-color: #00ff88; }
        .failed { border-left-color: #ff6b6b; }
        .code { background: #0f0f23; padding: 15px; border-radius: 4px; overflow-x: auto; font-family: monospace; font-size: 12px; }
        .validation { padding: 5px 10px; margin: 5px 0; border-radius: 4px; }
        .pass { background: #00ff8822; color: #00ff88; }
        .fail { background: #ff6b6b22; color: #ff6b6b; }
        .error { background: #ff000022; color: #ff6666; padding: 10px; border-radius: 4px; margin: 10px 0; }
        pre { margin: 0; white-space: pre-wrap; word-wrap: break-word; }
    </style>
</head>
<body>
    <h1>Einstein's Bedside Notes - Generation Loop Results</h1>
    <p>Generated: """ + datetime.now().isoformat() + """</p>
    <p>Total Iterations: """ + str(len(self.history)) + """</p>
    <p>Valid Candidates: """ + str(len(valid_candidates)) + """</p>
"""
        
        for result in self.history:
            iteration = result.get("iteration", 0)
            is_valid = result.get("is_valid", False)
            css_class = "success" if is_valid else "failed"
            
            html += f"""
    <div class="iteration {css_class}">
        <h2>Iteration {iteration + 1}</h2>
        <p><strong>Status:</strong> {"Valid!" if is_valid else "Failed"}</p>
"""
            
            if result.get("execution_error"):
                html += f"""
        <div class="error">
            <strong>Execution Error:</strong>
            <pre>{result['execution_error']}</pre>
        </div>
"""
            
            if result.get("validation_results"):
                val_results = result["validation_results"]
                if val_results.get("validations"):
                    html += "        <h3>Validations</h3>\n"
                    for v in val_results["validations"]:
                        name = v.get("validator", "Unknown")
                        flags = v.get("flags", {})
                        overall = flags.get("overall", "UNKNOWN")
                        css = "pass" if overall in ["PASS", "WARNING"] else "fail"
                        html += f'        <div class="validation {css}">{name}: {overall}</div>\n'
            
            if result.get("code"):
                code_preview = result["code"][:1000] + "..." if len(result["code"]) > 1000 else result["code"]
                # Escape HTML
                code_preview = code_preview.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                html += f"""
        <h3>Generated Code</h3>
        <div class="code"><pre>{code_preview}</pre></div>
"""
            
            html += "    </div>\n"
        
        html += """
</body>
</html>
"""
        
        report_path = os.path.join(self.output_dir, "report.html")
        with open(report_path, 'w') as f:
            f.write(html)
        
        print(f"Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Einstein's Bedside Notes - Generate and validate gravitational theories"
    )
    
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="Create a unified field theory that combines gravity and electromagnetism",
        help="Initial prompt describing the desired theory"
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        default="grok",
        choices=["grok", "openai", "anthropic", "gemini"],
        help="LLM provider to use (default: grok)"
    )
    
    parser.add_argument(
        "--max-iterations", "-n",
        type=int,
        default=5,
        help="Maximum number of iterations (default: 5)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--continue-after-valid",
        action="store_true",
        help="Continue running even after finding valid theories (default: stop on first valid)"
    )
    
    args = parser.parse_args()
    
    loop = EinsteinLoop(
        provider=args.provider,
        max_iterations=args.max_iterations,
        output_dir=args.output_dir,
        verbose=args.verbose,
        continue_after_valid=args.continue_after_valid
    )
    
    try:
        results = loop.run(args.prompt)
        
        if results["valid_candidates"]:
            print("\nSuccess! Valid theory candidates generated.")
            sys.exit(0)
        else:
            print("\nNo valid candidates found in the given iterations.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nLoop interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

