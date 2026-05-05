#!/usr/bin/env python3
"""
SWE Task Validator - Validates trajectory outcomes against task requirements.

This module provides task-aware validation by comparing:
1. The actual changes made (from patch.diff)
2. The task requirements (from evaluation platform-benchmark.yaml or task description)

This is used as a final validation step after structural matching.
"""

import os
import json
import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TaskValidationResult:
    """Result of task validation."""
    is_valid: bool
    score: float  # 0.0 to 1.0
    reasoning: str
    issues: List[str]
    task_description: str
    diff_summary: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "score": self.score,
            "reasoning": self.reasoning,
            "issues": self.issues,
            "task_description": self.task_description[:200] + "..." if len(self.task_description) > 200 else self.task_description,
            "diff_summary": self.diff_summary[:500] + "..." if len(self.diff_summary) > 500 else self.diff_summary
        }


class SWETaskValidator:
    """
    Validates trajectory outcomes against task requirements.
    
    Uses LLM to compare:
    - The actual diff produced by the agent
    - The task requirements
    
    This helps distinguish between:
    - Correct implementations (pass)
    - Structurally similar but incorrect implementations (fail)
    """
    
    def __init__(self, llm_prefix: str = "AZURE"):
        self.llm_prefix = llm_prefix
    
    def validate_trajectory(
        self, 
        trajectory_dir: str,
        task_description: Optional[str] = None
    ) -> TaskValidationResult:
        """
        Validate a trajectory against its task requirements.
        
        Args:
            trajectory_dir: Path to trajectory directory (containing output/vsc-output/)
            task_description: Optional task description (will be loaded if not provided)
            
        Returns:
            TaskValidationResult with validation outcome
        """
        # Find the vsc-output directory
        vsc_output = self._find_vsc_output(trajectory_dir)
        if not vsc_output:
            return TaskValidationResult(
                is_valid=False,
                score=0.0,
                reasoning="Could not find vsc-output directory",
                issues=["Missing vsc-output"],
                task_description="",
                diff_summary=""
            )
        
        # Load patch.diff
        patch_diff = self._load_patch_diff(vsc_output)
        if not patch_diff:
            return TaskValidationResult(
                is_valid=False,
                score=0.0,
                reasoning="No patch.diff found - no changes made",
                issues=["No changes"],
                task_description="",
                diff_summary=""
            )
        
        # Load task description if not provided
        if not task_description:
            task_description = self._load_task_description(vsc_output)
        
        if not task_description:
            return TaskValidationResult(
                is_valid=False,
                score=0.0,
                reasoning="Could not load task description",
                issues=["Missing task description"],
                task_description="",
                diff_summary=patch_diff[:500]
            )
        
        # Use LLM to validate the diff against task requirements
        return self._validate_with_llm(patch_diff, task_description)
    
    def _find_vsc_output(self, trajectory_dir: str) -> Optional[str]:
        """Find the vsc-output directory."""
        # Try common paths
        candidates = [
            os.path.join(trajectory_dir, "output", "vsc-output"),
            os.path.join(trajectory_dir, "vsc-output"),
            trajectory_dir
        ]
        
        for path in candidates:
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "chat-export-logs.json")):
                return path
        
        return None
    
    def _load_patch_diff(self, vsc_output: str) -> Optional[str]:
        """Load the patch.diff file."""
        patch_path = os.path.join(vsc_output, "patch.diff")
        if os.path.exists(patch_path):
            with open(patch_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        return None
    
    def _load_task_description(self, vsc_output: str) -> Optional[str]:
        """Load task description from config files."""
        # Try evaluation platform-benchmark.yaml first
        benchmark_path = os.path.join(vsc_output, "configs", "evaluation platform-benchmark.yaml")
        if os.path.exists(benchmark_path):
            with open(benchmark_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Extract the task text from promptSteps
                match = re.search(r'text:\s*\n?\s*(.+?)(?=\n\s*assertions:|$)', content, re.DOTALL)
                if match:
                    return match.group(1).strip()
                # Try simpler extraction
                match = re.search(r'text:\s*(.+?)(?:\n\s+\w+:|$)', content, re.DOTALL)
                if match:
                    return match.group(1).strip()
        
        # Try task.md
        task_md_path = os.path.join(vsc_output, "configs", "task.md")
        if os.path.exists(task_md_path):
            with open(task_md_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        return None
    
    def _validate_with_llm(
        self, 
        patch_diff: str, 
        task_description: str
    ) -> TaskValidationResult:
        """Use LLM to validate the diff against task requirements."""
        try:
            try:
                from llm import get_model_and_client
            except ImportError:
                from src.llm import get_model_and_client
        except ImportError:
            logger.warning("LLM module not available")
            return self._validate_heuristic(patch_diff, task_description)
        
        # Truncate diff if too long
        max_diff_len = 4000
        if len(patch_diff) > max_diff_len:
            patch_diff = patch_diff[:max_diff_len] + "\n... (truncated)"
        
        prompt = self._build_validation_prompt(patch_diff, task_description)
        
        try:
            client, model, temp = get_model_and_client(self.llm_prefix)
            
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert code reviewer validating if code changes correctly implement a task.

Your job is to determine if the diff (code changes) correctly and completely implements the task requirements.

A VALID implementation must:
1. Make all required changes described in the task
2. NOT break existing functionality 
3. NOT leave old code that should be removed
4. Produce syntactically correct code

Common issues that make an implementation INVALID:
- Adding new code without removing old code (duplicates)
- Incomplete renaming (missing some references)
- Breaking syntax (missing colons, indentation errors, etc.)
- Not updating all required files

Respond with EXACTLY a JSON object:
{
    "is_valid": true/false,
    "score": 0.0-1.0,
    "reasoning": "brief explanation of validity",
    "issues": ["list", "of", "specific", "issues"] 
}

If is_valid is true, issues should be empty or contain minor warnings.
If is_valid is false, issues should list specific problems found."""
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            result = self._parse_validation_response(content)
            
            if result:
                return TaskValidationResult(
                    is_valid=result["is_valid"],
                    score=result["score"],
                    reasoning=result["reasoning"],
                    issues=result["issues"],
                    task_description=task_description,
                    diff_summary=patch_diff[:500]
                )
                
        except Exception as e:
            logger.warning(f"LLM validation failed: {e}")
        
        return self._validate_heuristic(patch_diff, task_description)
    
    def _build_validation_prompt(self, patch_diff: str, task_description: str) -> str:
        """Build the validation prompt."""
        return f"""=== TASK REQUIREMENTS ===
{task_description}

=== CODE CHANGES (DIFF) ===
{patch_diff}

=== QUESTION ===
Do the code changes correctly and completely implement the task requirements?
Look for issues like:
- Duplicate definitions (old and new code both present)
- Missing renames/updates
- Syntax errors in the changes
- Incomplete changes (some files/references not updated)

Respond with JSON: {{"is_valid": true/false, "score": 0.0-1.0, "reasoning": "...", "issues": [...]}}"""
    
    def _parse_validation_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse the LLM validation response."""
        try:
            # Find JSON object in response
            match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if match:
                result = json.loads(match.group())
                return {
                    "is_valid": bool(result.get("is_valid", False)),
                    "score": float(result.get("score", 0.5)),
                    "reasoning": str(result.get("reasoning", "")),
                    "issues": list(result.get("issues", []))
                }
        except Exception as e:
            logger.debug(f"Failed to parse validation response: {e}")
        return None
    
    def _validate_heuristic(
        self, 
        patch_diff: str, 
        task_description: str
    ) -> TaskValidationResult:
        """Heuristic validation when LLM is not available."""
        issues = []
        score = 1.0
        
        # Check for duplicate function definitions (common in failed refactoring)
        lines = patch_diff.split('\n')
        func_defs_in_same_section = []
        current_file = ""
        
        for line in lines:
            if line.startswith('diff --git') or line.startswith('+++'):
                current_file = line
                func_defs_in_same_section = []
            elif line.startswith('+') and 'def ' in line and not line.startswith('+++'):
                # Found a function definition being added
                func_name = re.search(r'def\s+(\w+)', line)
                if func_name:
                    func_defs_in_same_section.append(func_name.group(1))
        
        # Check for empty lines followed by new function defs (sign of duplication)
        if '\n+\n+def ' in patch_diff:
            issues.append("Possible duplicate function definitions (empty line + new def)")
            score -= 0.3
        
        # Check if the diff looks like proper replacement vs addition
        additions = len([l for l in lines if l.startswith('+') and not l.startswith('+++')])
        deletions = len([l for l in lines if l.startswith('-') and not l.startswith('---')])
        
        if additions > 0 and deletions == 0:
            issues.append("Only additions, no deletions - may be incomplete refactoring")
            score -= 0.2
        
        is_valid = len(issues) == 0 and score > 0.5
        
        return TaskValidationResult(
            is_valid=is_valid,
            score=max(0, score),
            reasoning="Heuristic validation" + (f" - found {len(issues)} issues" if issues else " - looks OK"),
            issues=issues,
            task_description=task_description,
            diff_summary=patch_diff[:500]
        )


def validate_trajectory_batch(
    trajectory_dirs: List[str],
    task_description: Optional[str] = None,
    llm_prefix: str = "AZURE"
) -> List[Tuple[str, TaskValidationResult]]:
    """
    Validate a batch of trajectories.
    
    Args:
        trajectory_dirs: List of trajectory directory paths
        task_description: Optional shared task description
        llm_prefix: LLM config prefix
        
    Returns:
        List of (trajectory_name, result) tuples
    """
    validator = SWETaskValidator(llm_prefix=llm_prefix)
    results = []
    
    for traj_dir in trajectory_dirs:
        name = os.path.basename(traj_dir)
        result = validator.validate_trajectory(traj_dir, task_description)
        results.append((name, result))
        
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate trajectory outcomes against task requirements")
    parser.add_argument("trajectory_dir", help="Path to trajectory directory")
    parser.add_argument("--task", "-t", help="Task description (optional)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    validator = SWETaskValidator()
    result = validator.validate_trajectory(args.trajectory_dir, args.task)
    
    print("\n" + "=" * 60)
    print("TASK VALIDATION RESULT")
    print("=" * 60)
    print(f"Valid: {result.is_valid}")
    print(f"Score: {result.score:.1%}")
    print(f"Reasoning: {result.reasoning}")
    if result.issues:
        print(f"Issues:")
        for issue in result.issues:
            print(f"  - {issue}")
