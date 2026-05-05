# PTA Matcher

Compare trajectories against a merged PTA (domtree) to measure coverage, validate execution paths, and predict pass/fail outcomes.

## Overview

The PTA Matcher compares individual coding agent trajectories against a merged reference PTA (domtree). It uses a **hybrid approach** combining:

- **Structural Coverage**: State machine subsequence matching
- **Process Validation**: Required tool sequence checking
- **Task Validation**: LLM-based content correctness analysis

This enables:

- **Pass/Fail Prediction**: Classify trajectories based on multiple signals
- **Validation**: Check if a trajectory follows the expected solution pattern
- **Analysis**: Understand where and why a trajectory diverged
- **Quality Metrics**: Quantify coverage as a success indicator

## Usage

### Single Comparison

```bash
python swe_pta_matcher.py <domtree.json> <trajectory.json> [options]
```

**Examples:**

```bash
# Basic comparison
python swe_pta_matcher.py outputs/merged_pta.json outputs/trajectory_pta.json

# JSON output
python swe_pta_matcher.py outputs/merged_pta.json outputs/trajectory_pta.json --json

# Verbose mode (debug)
python swe_pta_matcher.py outputs/merged_pta.json outputs/trajectory_pta.json --verbose

# Without LLM (heuristics only, faster)
python swe_pta_matcher.py outputs/merged_pta.json outputs/trajectory_pta.json --no-llm
```

### Batch Comparison

```bash
python swe_pta_matcher.py <domtree.json> --batch <traj1.json> <traj2.json> ...
```

**Example:**

```bash
python swe_pta_matcher.py outputs/merged_pta.json --batch \
    outputs/run1_pta.json \
    outputs/run2_pta.json \
    outputs/run3_pta.json
```

### Incremental Mode

For streaming/real-time comparison:

```bash
# First buffer
python swe_pta_matcher.py domtree.json buffer1.json --offset 0
# → PASS, OFFSET: 3

# Next buffer (continues from offset 3)
python swe_pta_matcher.py domtree.json buffer2.json --offset 3
# → PASS, OFFSET: 6
```

## CLI Options

| Flag | Description |
|------|-------------|
| `--batch` | Compare multiple trajectories |
| `--json` | Output JSON format |
| `--verbose`, `-v` | Debug output |
| `--offset N` | Incremental mode starting at position N |
| `--no-llm` | Disable LLM equivalence (use heuristics only) |
| `--no-task-validation` | Disable task validation (skip diff checking) |
| `--llm-prefix` | Environment variable prefix for LLM config |

## Core Concepts

### Subsequence Coverage

Measures how many domtree states appear in the trajectory **in the same relative order**.

```
Domtree:     A → B → C → D
Trajectory:  A → X → B → Y → C → D

Coverage = 100% (A, B, C, D all appear in order)
Extra states X, Y are allowed (agent took additional steps)
```

**Counter-example:**

```
Domtree:     A → B → C → D
Trajectory:  A → C → B → D  (B and C swapped)

Coverage < 100% because once we match A then C, we can't go back to match B.
```

### Terminal State Match

Whether the trajectory ends in the same final state as the domtree.

- **Terminal match = True**: Trajectory achieved the goal (even if path differed)
- **Terminal match = False**: Trajectory ended in a different state

### Tree-Structured Domtrees

When merging multiple successful trajectories, the domtree may have branches (different valid approaches):

```
Domtree:
    A → B → C → D    (Path 1: search first)
     \→ E → F → D    (Path 2: create directly)
```

The matcher tries **all paths** and reports the **best match**. A trajectory following Path 2 will get 100% coverage against Path 2.

## Interpreting Results

### Verdict System

The matcher produces a **verdict** based on multiple signals:

| Verdict | Meaning |
|---------|---------|
| **PASS** | High coverage (≥80%) + all required tools present |
| **LIKELY PASS** | Good coverage (≥60%) + all tools present |
| **UNCERTAIN** | Mixed signals, manual review needed |
| **LIKELY FAIL** | Missing required tools or low coverage |
| **FAIL** | Task validation failed or very low coverage |

### Understanding the Metrics

| Metric | What It Measures |
|--------|------------------|
| **Coverage %** | How many domtree states appear in trajectory (in order) |
| **Process Coverage %** | How many required tools the trajectory used |
| **Task Valid** | Whether the actual code changes are correct (LLM-based) |
| **Terminal Match** | Whether trajectory ends in same state as domtree |

### Decision Matrix

| Coverage | Process | Task Valid | Verdict |
|----------|---------|------------|---------|
| ≥80% | 100% | VALID | **PASS** |
| ≥60% | 100% | VALID | **LIKELY PASS** |
| Any | <100% | - | **LIKELY FAIL** (missing tools) |
| Any | - | INVALID | **FAIL** (content errors) |
| <50% | <100% | - | **FAIL** |

### Example Output

```
================================================================================
BATCH COMPARISON SUMMARY
================================================================================

Process Validation:
  Required tools from domtree: ['create_file', 'run_in_terminal', 'open_simple_browser']

Missing tools (process validation):
  gpt-4.1-fail: run_in_terminal
  gpt-5-mini-fail: run_in_terminal, open_simple_browser

Detailed results:
--------------------------------------------------------------------------------
Trajectory                        Coverage  Process  TaskValid  Verdict
--------------------------------------------------------------------------------
gemini-flash-pass                   100.0%    100%     VALID    PASS
gpt-5.1-codex-pass                  100.0%    100%     VALID    PASS
gpt-4.1-fail                         33.3%     67%     VALID    LIKELY FAIL
gpt-5-mini-fail                      50.0%     33%     VALID    LIKELY FAIL
--------------------------------------------------------------------------------
```

## Validation Layers

### Layer 1: Structural Coverage

Measures how many domtree states appear in the trajectory **in the same relative order**.

```
Domtree:     A → B → C → D
Trajectory:  A → X → B → Y → C → D

Coverage = 100% (A, B, C, D all appear in order)
```

### Layer 2: Process Validation

Checks if the trajectory used all **required tools** from the domtree.

Required tools = tools that appear in **all successful paths**:

```
Required: {create_file, run_in_terminal, open_simple_browser}
Trajectory has: {create_file, open_simple_browser}
Missing: {run_in_terminal}
Process Coverage: 67%
```

### Layer 3: Task Validation

Uses LLM to analyze actual code changes (`patch.diff`) against task requirements:

- Does the diff accomplish the core task?
- Are there syntax errors or bugs?
- Is the implementation complete?

## State Equivalence

States are compared using multiple strategies:

1. **Position 0**: Initial states always match
2. **Content Hash**: Same content → equivalent
3. **Resulting State**: Same abstract outcome → equivalent
4. **Heuristic Match**: Tool-specific rules (file paths, queries)
5. **LLM Match**: Semantic comparison for ambiguous cases

## Python API

```python
from swe_pta_matcher import batch_compare, print_batch_summary

results = batch_compare(
    domtree_path="merged_pta.json",
    trajectory_paths=["traj1.json", "traj2.json"],
    use_llm=True,
    run_task_validation=True
)

print_batch_summary(results)

for r in results:
    print(f"{r['trajectory']}: {r['verdict']}")
```

## Why Multiple Layers?

Pure structural coverage is **insufficient** for coding agents:

| Limitation | Example |
|------------|---------|
| **Process Failures** | Skips starting server but opens browser |
| **Content Failures** | Uses correct tools but writes buggy code |

The hybrid approach catches both failure types.
