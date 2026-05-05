# SWE Agent Trajectory Analysis Tutorial

This tutorial demonstrates how to use the SWE PTA (Prefix Tree Acceptor) system to analyze and validate coding agent behavior by learning from execution trajectories.

## Overview

The system:
1. **Captures** coding agent trajectories (tool calls, file operations, LLM responses)
2. **Generates** PTAs (state machines) from each trajectory
3. **Merges** PTAs to identify common patterns and variations
4. **Visualizes** the results to understand agent behavior

---

## Data Structure

### Required Folder Structure

```
coding-agent-trajectories/
├── run-12345-instance-task-name-logs/
│   └── output/
│       └── vsc-output/
│           └── chat-export-logs.json    ← Main trajectory file
├── run-67890-instance-task-name-logs/
│   └── output/
│       └── vsc-output/
│           └── chat-export-logs.json
└── ...
```

Each instance folder contains a `chat-export-logs.json` file with the agent's execution trace.

### chat-export-logs.json Format

```json
{
  "exportedAt": "2026-01-22T12:27:30.583Z",
  "totalPrompts": 1,
  "totalLogEntries": 3,
  "prompts": [
    {
      "prompt": "Create a petstore API...",
      "logs": [
        {
          "id": "abc123",
          "kind": "request",           // LLM request/response
          "metadata": {
            "model": "gpt-5.2",
            ...
          },
          "response": "I'll create..."
        },
        {
          "id": "call_xyz",
          "kind": "toolCall",          // Tool invocation
          "function": {
            "name": "create_file",
            "arguments": "{\"filePath\": \"/workspace/app.py\", ...}"
          },
          "response": "File created successfully"
        },
        ...
      ]
    }
  ]
}
```

**Key entry types:**
- `kind: "request"` - LLM request/response pairs
- `kind: "toolCall"` - Tool invocations (create_file, read_file, run_in_terminal, etc.)

---

## Phase 1: Generate PTA from Single Trajectory

Generate a PTA from a single trajectory to understand its structure:

```bash
python extract_swe_ground_truth.py ./coding-agent-trajectories \
  "run-21248319029-instance-chat_mode_simple-logs" \
  --only-generate-pta \
  --output-dir ./swe_outputs \
  --verbose
```

**Output:**
```
Processing instance: run-21248319029-instance-chat_mode_simple-logs
Found trajectory: .../chat-export-logs.json
Generated PTA with 5 states and 4 transitions
Saved: swe_outputs/run-21248319029-instance-chat_mode_simple-logs_pta.json
```

### Understanding the Generated PTA

```json
{
  "initial_state": "state_0",
  "states": {
    "state_0": {
      "state_id": "state_0",
      "step": 0,
      "observation": "<initial>",
      "tool_used": null
    },
    "state_1": {
      "state_id": "state_1",
      "step": 2,
      "observation": "LLM Request: gpt-5.2\nResponse: I'll create SPEC.md...",
      "tool_used": null,
      "log_entry": { "kind": "request", "model": "gpt-5.2", ... }
    },
    "state_2": {
      "state_id": "state_2",
      "step": 3,
      "observation": "Tool: create_file\nfilePath: /workspace/SPEC.md\n...",
      "tool_used": "create_file",
      "files_touched": ["/workspace/SPEC.md"],
      "log_entry": { "kind": "toolCall", "tool": "create_file", ... }
    }
  },
  "transitions": [
    {
      "from_state": "state_0",
      "to_state": "state_1",
      "action_type": "request"
    },
    {
      "from_state": "state_1",
      "to_state": "state_2",
      "action_type": "create_file"
    }
  ]
}
```

**State represents:**
- An LLM request/response, OR
- A tool call with its result

**Transitions represent:**
- The action that moves from one state to the next

---

## Phase 2: Visualize Single PTA

Use the visualizer to see the PTA structure:

```bash
# Generate all visualizations
python swe_pta_visualizer.py ./swe_outputs/run-21248319029-instance-chat_mode_simple-logs_pta.json --all
```

**Output files:**
- `*_visualization.txt` - Text statistics and flow diagram
- `*_graph.html` - Interactive DAG visualization (open in browser)
- `*_list.html` - Linear transition list

### Text Output Example

```
SWE PTA STATISTICS SUMMARY
============================================================

📊 Basic Counts:
  Total States: 5
  Total Transitions: 4
  Initial State: state_0
  Terminal States: 1

🔧 Tool/Action Types:
  request: 2 (50.0%)
  create_file: 2 (50.0%)

📁 Files Touched: 1
  /workspace/SPEC.md
```

### Interactive Graph View

Open `*_graph.html` in a browser to see:
- Color-coded nodes by tool type
- Draggable layout
- Branching/merging points highlighted
- Stats sidebar

---

## Phase 3: Generate and Merge Multiple PTAs

Process multiple trajectories for the same task to identify common patterns:

```bash
python extract_swe_ground_truth.py ./coding-agent-trajectories \
  "run-21248319029-instance-chat_mode_simple-logs,run-21244897627-instance-chat_mode_simple-logs,run-21238393712-instance-chat_mode_simple-logs" \
  --output-dir ./swe_outputs \
  --verbose
```

**What happens:**
1. **Generate PTAs** - Creates a PTA for each trajectory
2. **Merge PTAs** - Combines them using state equivalence detection
3. **Output** - Merged PTA showing common structure + variations

**Output:**
```
Phase 1: Generating PTAs...
  Generated: run-21248319029-instance-chat_mode_simple-logs_pta.json (5 states)
  Generated: run-21244897627-instance-chat_mode_simple-logs_pta.json (6 states)
  Generated: run-21238393712-instance-chat_mode_simple-logs_pta.json (5 states)

Phase 2: Merging PTAs...
  Merging 3 PTAs...
  Merged PTA: 7 states, 6 transitions
  Branches created: 1 (traces diverged at step 2)

Saved: swe_outputs/merged_pta.json
```

---

## Phase 4: Understanding the Merged PTA

The merged PTA shows where traces are similar and where they diverge:

```json
{
  "initial_state": "merged_state_5",
  "states": {
    "merged_state_5": {
      "observation": "<initial>",
      "metadata": {
        "original_ids": ["merged_state_1", "state_0"],
        "trace_count": 2
      }
    },
    "merged_state_6": {
      "observation": "LLM Request: gpt-5.2...",
      "metadata": {
        "trace_count": 1          // Only 1 trace went this path
      }
    },
    "merged_state_9": {
      "observation": "LLM Request: gpt-5.2...",  // Different response
      "metadata": {
        "trace_count": 1,
        "branch_from": "merged_state_6"   // Branched here
      }
    }
  },
  "branches": {
    "merged_state_6": ["merged_state_9", ...]  // Branch points
  },
  "metadata": {
    "merge_stats": {
      "traces_merged": 2,
      "states_merged": 5,
      "branches_created": 1
    }
  }
}
```

**Key indicators:**
- `trace_count` - How many traces passed through this state
- `branch_from` - Where a branch originated
- `branches` - Map of branch points to alternative paths

---

## Phase 5: Visualize Merged PTA

```bash
python swe_pta_visualizer.py ./swe_outputs/merged_pta.json --all
```

**The graph visualization shows:**
- 🟢 **Green border** - Initial state
- 🔴 **Red border** - Terminal states  
- 🔵 **Blue border** - Branching points (out-degree > 1)
- 🟠 **Orange border** - Merge points (in-degree > 1)

**Statistics include merge info:**
```
🔀 Merge Statistics:
  Traces Merged: 2
  States Added: 11
  States Merged: 5
  Branches Created: 1
```

---

## Phase 6: Compare Trajectories Against Merged PTA

Once you have a merged PTA (domtree) from successful trajectories, you can compare new trajectories against it to predict pass/fail outcomes.

The **PTA Matcher** uses a hybrid approach combining:
- **Structural Coverage** - How well the trajectory follows the domtree pattern
- **Process Validation** - Whether all required tools were used
- **Task Validation** - Whether the actual code changes are correct

```bash
# Compare multiple trajectories against the merged reference
python swe_pta_matcher.py ./swe_outputs/merged_pta.json --batch \
    ./swe_outputs/trajectory1_pta.json \
    ./swe_outputs/trajectory2_pta.json \
    ./swe_outputs/trajectory3_pta.json
```

**Example output:**
```
Process Validation:
  Required tools from domtree: ['create_file', 'run_in_terminal', 'open_simple_browser']

Detailed results:
--------------------------------------------------------------------------------
Trajectory                        Coverage  Process  TaskValid  Verdict
--------------------------------------------------------------------------------
trajectory1_pta.json                100.0%    100%     VALID    PASS
trajectory2_pta.json                 33.3%     67%     VALID    LIKELY FAIL
--------------------------------------------------------------------------------
```

📖 **For full details on interpreting results and all CLI options, see [PTA_MATCHER.md](PTA_MATCHER.md).**

---

## State Equivalence

When merging, the system determines if two states are "equivalent" using a 3-tier approach:

### Tier 1: Exact Match
- Same tool used
- Same observation hash → **Equivalent**

### Tier 2: Heuristic Match
Tool-specific rules:
- `create_file`: Same file path → Equivalent (content may differ)
- `read_file`: Same file path → Equivalent
- `file_search`/`grep_search`: Similar query → Equivalent
- `run_in_terminal`: Same base command → Equivalent

### Tier 3: LLM Semantic Match
For ambiguous cases, uses LLM to compare:
- Are these states functionally equivalent?
- Is this the same logical step in the task?

---

## Quick Reference

### Generate Single PTA
```bash
python extract_swe_ground_truth.py ./coding-agent-trajectories \
  "<instance-id>" --only-generate-pta --output-dir ./swe_outputs
```

### Generate & Merge Multiple PTAs
```bash
python extract_swe_ground_truth.py ./coding-agent-trajectories \
  "<id1>,<id2>,<id3>" --output-dir ./swe_outputs
```

### Process All Instances
```bash
python extract_swe_ground_truth.py ./coding-agent-trajectories \
  --all --output-dir ./swe_outputs
```

### Visualize PTA
```bash
# All formats
python swe_pta_visualizer.py ./swe_outputs/merged_pta.json --all

# Just interactive graph
python swe_pta_visualizer.py ./swe_outputs/merged_pta.json --html

# Just text stats
python swe_pta_visualizer.py ./swe_outputs/merged_pta.json
```

---

## Example Workflow

```bash
# 1. Generate individual PTAs to understand each trace
python extract_swe_ground_truth.py ./coding-agent-trajectories \
  "run-21248319029-instance-chat_mode_simple-logs" \
  --only-generate-pta --output-dir ./swe_outputs

# 2. Visualize the individual PTA
python swe_pta_visualizer.py ./swe_outputs/run-21248319029-instance-chat_mode_simple-logs_pta.json --html

# 3. Merge multiple traces for the same task
python extract_swe_ground_truth.py ./coding-agent-trajectories \
  "run-21248319029-instance-chat_mode_simple-logs,run-21244897627-instance-chat_mode_simple-logs" \
  --output-dir ./swe_outputs

# 4. Visualize the merged PTA to see patterns
python swe_pta_visualizer.py ./swe_outputs/merged_pta.json --all

# 5. Open the graph HTML in browser
# → swe_outputs/merged_pta_graph.html
```

---

## Tool Color Legend (in visualizations)

| Tool | Color | Description |
|------|-------|-------------|
| `create_file` | 🟢 Green | File creation |
| `replace_string_in_file` | 🟡 Yellow | File editing |
| `read_file` | 🔵 Blue | File reading |
| `file_search` | 🟣 Purple | File search |
| `grep_search` | 🩷 Pink | Text search |
| `semantic_search` | 🟠 Orange | Semantic search |
| `run_in_terminal` | 🔴 Red | Terminal commands |
| `list_dir` | 🩵 Teal | Directory listing |
| `request` | ⚪ Gray | LLM request/response |

---

## Troubleshooting

### "No trajectory file found"
- Check folder structure matches expected: `instance/output/vsc-output/chat-export-logs.json`
- The system searches recursively, but verify the file exists

### "All traces branch immediately"
- Different LLM models cause branches (model is part of signature)
- Different response content causes branches
- This is expected for diverse executions

### Empty visualization
- Verify the JSON is a PTA file (has `states` and `transitions` keys)
- Raw `chat-export-logs.json` must be processed first with `extract_swe_ground_truth.py`
