# AgentLens User Guide

**AgentLens** is a web-based tool for analyzing, profiling, and comparing AI coding agent trajectories. It helps you understand how an agent solved a task, assess its quality against a ground truth, detect inefficiencies, and compare multiple agents side by side.

**Sample data**: We've included sample ATIF data you can use to try out AgentLens right away. It's located at:
[`data/atif_data/`](../data/atif_data/).

> Authentication is required to access the hosted version.

---

## Table of Contents

1. [Preparing Your Data](#1-preparing-your-data)
2. [Uploading Trajectories](#2-uploading-trajectories)
3. [Viewing the Instance Table](#3-viewing-the-instance-table)
4. [Behavioral Profile (Tier 1)](#4-behavioral-profile-tier-1)
5. [Quality Assessment (Tier 2)](#5-quality-assessment-tier-2)
6. [Comparing Trajectories](#6-comparing-trajectories)
7. [LLM-Powered Analysis](#7-llm-powered-analysis)
8. [Tips & FAQ](#8-tips--faq)

---

## 1. Preparing Your Data

AgentLens accepts trajectory data in several formats. The most feature-rich experience is with **ATIF** (Agent Trace Interchange Format) data.

### ATIF ZIP (Recommended)

An ATIF ZIP file is a session recording from a coding agent. The ZIP contains a main `trajectory.json` and optionally subagent trajectory files. Two layouts are supported:

**Standard layout** (with `atif/` subfolder):
```
my_session.zip
└── atif/
    └── copilot/
        ├── trajectory.json          ← main agent trajectory
        ├── subagent-001.json        ← 1st subagent trajectory (optional)
        └── subagent-002.json        ← 2nd subagent trajectory (optional)
```

**Flat layout** (no nesting):
```
my_session.zip
├── trajectory.json                  ← main agent trajectory
├── subagent-001.json                ← 1st subagent trajectory (optional)
└── subagent-002.json                ← 2nd subagent trajectory (optional)
```

Subagent files are placed **in the same directory** as the main `trajectory.json` and follow the naming convention `subagent-NNN.json` (zero-padded, e.g., `subagent-001.json`). The main trajectory references them via a `subagent_trajectory_ref` field, and AgentLens automatically inlines the subagent steps when loading.

> Not all sessions have subagents — this depends on the agent. For example, Copilot sessions may spawn subagents for delegated tasks, while Cursor sessions typically don't.

Each JSON file has a `schema_version` field starting with `"ATIF"` and contains:

- **Agent metadata**: agent name, model, task/scenario name
- **Step-by-step actions**: tool calls, file edits, terminal commands, etc.
- **Human Experience (HX) data**: wall time, active time, permission waits, human inputs, LLM latencies, token usage, subagent spawns, and context compactions

> **Why ATIF?** ATIF data unlocks the full feature set — Human Experience metrics, time breakdowns, latency charts, token pressure graphs, HX Score, and more. Other formats show core behavioral analysis but don't include HX metrics.

### Other Supported Formats

| Format | What to upload | Notes |
|--------|---------------|-------|
| **Agent Log ZIP** | ZIP containing `chat-export-logs.json` (usually under `output/vsc-output/`) | VS Code chat export logs |
| **OpenHands ZIP/JSON** | ZIP or JSON file with OpenHands trajectory data | Fallback format |
| **Pre-saved SDK Trace JSON** | A `.json` file with an `initial_state` field | Previously processed trace |

### Pass/Fail Detection

AgentLens tries to auto-detect whether a trajectory passed or failed:

- From an `eval.json` inside the ZIP (looks for a `resolved` or `passed` field)
- From the filename: files containing `-pass-` are marked passing, `-fail-` are marked failing
- You can also manually toggle pass/fail status in the app after uploading

### Organizing Files for Comparison

If you plan to **compare agents** or **build a ground truth**, prepare multiple session ZIPs for the **same task**. For example:

```
piececolor_enum_refactor/
├── claude_session1.zip      (pass)
├── claude_session2.zip      (pass)
├── copilot_session1.zip     (fail)
├── cursor_session1.zip      (pass)
└── cursor_session2.zip      (pass)
```

<!-- ### Sample Data

We've included sample ATIF data you can use to try out AgentLens right away. It's located at:

```
data/atif_data/
├── piececolor_enum_refactor/       (7 sessions across Claude, Copilot, Cursor)
└── table_connection_string_loopback/
```

Upload any of the ZIP files from these folders (relative to the repo root: [`data/atif_data/`](../data/atif_data/)) to get started. -->

---

## 2. Uploading Trajectories

When you first open AgentLens, you'll see a landing page with a large upload area.


### Drag & Drop
Drag one or more `.zip` or `.json` files directly onto the upload zone.

### Browse Files
Click the upload zone to open a file picker. You can select multiple files at once.

### Upload Folder
Click the **Upload Folder** button to select an entire directory. All supported files in the folder will be uploaded.

After uploading, trajectories appear in the instance table within a few seconds.

> **Note**: Maximum upload size is 100 MB per file.

---

## 3. Viewing the Instance Table

Once you have trajectories loaded, the landing page shows a table with one row per trajectory:


| Column | Description |
|--------|-------------|
| **Benchmark** | Auto-detected benchmark name (e.g., SWE-bench) |
| **Task Name** | The task or scenario the agent was working on |
| **Instance** | Instance ID (clickable, sortable) |
| **Steps** | Number of steps in the trajectory |
| **Resolved** | Pass ✅ / Fail ❌ / Unknown ❓ |
| **Actions** | Icon buttons (see below) |

### Action Buttons (per row)

| Icon | Action |
|------|--------|
| 📥 | **Download** the raw trajectory JSON |
| 📄 | **View** the raw trajectory log |
| 📊 | **Open Behavioral Profile** — this is the main entry point for analysis |
| ✕ | **Delete** the trajectory |

### Header Controls

- **Include Scores**: Toggle to show/hide score columns
- **Upload**: Add more files without leaving the table
- **Export**: Export the table data
- **Stats bar**: Shows total instances and resolved count (e.g., "Resolved: 3/5 (60%)")

### Compare Button

When you have **2 or more trajectories** loaded, a floating **"Compare Trajectories"** button appears at the bottom-right corner.

---

## 4. Behavioral Profile (Tier 1)

Click the 📊 icon on any trajectory row to open its **Behavioral Profile**. This gives you a comprehensive view of how the agent worked.

### At a Glance — Stat Cards


A grid of key metrics appears at the top:

| Metric | What it means |
|--------|--------------|
| **States** | Total number of distinct states (actions) in the trace |
| **Files** | How many files the agent interacted with |
| **Tools** | How many different tools the agent used |
| **Coherence** | How cleanly the agent moved forward (0–100%). High = efficient, no backtracking. Low = the agent was thrashing or going in circles |
| **Completed** | Whether the agent reached a terminal state |
| **Explore / Implement** | Ratio of exploration steps vs. implementation steps |
| **Files Modified** | Number of files actually changed (vs. read-only) |
| **Human Inputs** | Number of times a human intervened (ATIF only) |
| **Subagents** | Number of sub-agents spawned (ATIF only) |
| **Active Time** | Total working time in seconds (ATIF only) |
| **Compactions** | Number of context window resets (ATIF only) |
| **HX Score** | Human Experience score 0–100 (ATIF only) — composite of Autonomy, Low Friction, Responsiveness, and Stability. Hover to see the breakdown |

### Human Experience Section (ATIF Only)

If your data is ATIF format, you get three additional visualizations:

- **Time Breakdown**: A stacked bar showing how time was split between Agent Work (green), LLM Thinking (blue), and Human Wait (orange)
- **Response Latency**: A sparkline chart of per-step LLM response times (ms), with an average line. Helps spot slowdowns
- **Token Pressure**: A chart of cumulative prompt tokens over time. Rising = context growing. Drops = compaction events (context window resets)

### Stage Distribution

A horizontal bar chart showing what proportion of steps fell into each stage:

- **Exploration** (blue) — Reading code, searching, understanding the codebase
- **Implementation** (green) — Writing code, making changes
- **Verification** (amber) — Running tests, checking results
- **Orchestration** (purple) — Planning, coordination, meta-actions

### Stage Timeline

A visual strip of colored blocks, one per step, showing the stage sequence over time. Human intervention points are marked with ▼ symbols. Below it is a **workflow fingerprint** — a compact string encoding of the stage pattern.

### Tools Used & Files Touched

Scrollable lists of every tool the agent called (with counts) and every file it interacted with.

### Navigation

- **"Back to Instances"** — return to the table
- **"Run Quality Assessment"** — proceed to Tier 2 assessment (see next section)

---

## 5. Quality Assessment (Tier 2)

Quality Assessment compares a trajectory against a **Ground Truth (GT)** — a merged reference built from multiple passing trajectories. This tells you *how well* the agent performed, not just *what* it did.

### Step 1: Provide Ground Truth


You have two options:

**Option A — Build GT from passing trajectories:**
1. Upload **at least 2 passing trajectory** ZIP files for the same task
2. Click **"Build GT & Assess"**
3. AgentLens merges them into a ground truth path-tree automaton (PTA) and assesses automatically

**Option B — Import a previously exported GT:**
1. Click **"Import merged GT JSON"**
2. Select a `merged_gt.json` file from a previous session
3. Assessment runs automatically

> **Tip**: After building a GT, click **"Export GT"** to save it as a JSON file. Next time you can skip uploading passing trajectories and just import the GT.

### Step 2: Review Results

The assessment results page has many sections. Here are the key ones:

#### Verdict & Quality Score
A large score card showing:
- **Quality Score** (0–100): overall quality rating
- **Verdict**: PASS, FAIL, PARTIAL, etc.
- **Quality Tier**: badge indicating the tier (e.g., "Gold", "Silver", "Bronze")
- **GT Info**: how many source traces and states the GT was built from

#### Key Metrics

| Metric | What it measures |
|--------|-----------------|
| **Coverage** | What percentage of the ground truth steps were matched |
| **Coherence** | How orderly the trajectory was |
| **Stage Completeness** | Whether all expected stages (explore → implement → verify) were present |
| **Workflow Similarity** | How closely the stage pattern matches the GT pattern |
| **F1 Score** | Balanced measure of precision and recall against GT |

#### Quality Signals

Color-coded cards highlighting important observations:
- 🔴 **Critical**: Serious issues (e.g., missed verification entirely)
- 🟡 **Warning**: Potential concerns (e.g., excessive retries)
- ℹ️ **Info**: Neutral observations

#### Divergence Points

Shows exactly where the trajectory diverged from the expected path — what the GT expected vs. what the agent actually did.

#### Failure Reasons

If the trajectory failed, this section explains **why** with detailed descriptions and severity levels.

#### Stage-Level Comparison

Per-stage breakdown showing:
- Expected steps (from GT)
- Matched steps
- Missing steps (in GT but not in trajectory)
- Extra steps (in trajectory but not in GT)

#### Inefficiency Analysis

Detailed breakdown of wasted effort:

| Inefficiency | Description |
|-------------|-------------|
| **Retry Loops** | Agent repeating the same action expecting different results |
| **Cyclic Patterns** | Agent going around in circles (A → B → C → A) |
| **Backtracks** | Agent undoing previous work |
| **Redundant Steps** | Unnecessary repeated actions |
| **Unnecessary Exploration** | Reading files that weren't relevant |

Includes a severity bar, per-tool breakdown table, total wasted steps, and wasted token count.

#### Visual Comparison

A **dual-lane visualization** showing the GT path (top) and the candidate trajectory (bottom) side by side:
- 🟢 Green blocks = matched steps
- 🔴 Red dashed blocks = GT steps that were missed
- 🟠 Orange blocks = extra steps not in GT
- Lines connect matching steps between the two lanes

#### GT Graph View

Click **"View GT"** to see the full ground truth as a directed acyclic graph (DAG):
- Nodes are colored by stage
- Branch points show where alternate paths exist
- Terminal nodes have red borders

#### Coverage Details

- **Process Coverage**: Were all required tools used? Lists any missing tools
- **File Coverage**: Were all required files touched? Lists any missing files

---

## 6. Comparing Trajectories

Compare up to 5 trajectories side by side to see which agent performed best.

### How to Compare

1. From the instance table, click the **"Compare Trajectories"** floating button (bottom-right)
2. Select **2 to 5 trajectories** from the list (click to toggle selection)
3. Click **"Compare N Trajectories"**
4. You'll be prompted to provide a Ground Truth (same options as Tier 2 — upload passing files or import GT JSON)

### Comparison Results

#### Metrics Table

A side-by-side table comparing all trajectories across three categories:

**Quality & Coverage:**
Quality Score, Structural Coverage, Coherence, Stage Completeness, Workflow Similarity, Process Coverage, File Coverage, Total Steps

**Inefficiency:**
Wasted Steps, Wasted Tokens, Retry Loops, Cyclic Patterns, Backtracks

**Human Experience (ATIF only):**
HX Score, Human Inputs, Active Time, Wall Time, Permission Wait, Subagents, Compactions

Best values in each row are **bolded** for easy comparison.


#### Per-Stage Effort

Shows how much effort each trajectory spent on each stage relative to the GT:
- **0×** = stage skipped entirely
- **1×** = ideal (matched GT exactly)
- **>1×** = overworked (spent more effort than expected)

#### GT Path Strategy

Use the dropdown to switch between:
- **Best Match (optimistic)**: Picks the GT path that maximizes each candidate's score individually
- **Canonical (fixed)**: Picks the single longest GT path with the most stages — same path for all candidates

#### GT Coverage Overlay

Toggle this on to see the GT graph with colored dots showing which candidates matched each node. Quickly identify which agent covered which parts of the solution.

---

## 7. LLM-Powered Analysis

AgentLens can send trajectory data to an LLM for deeper analysis. These features require the app to be configured with an LLM provider.

### LLM Assessment (from Quality Assessment view)

Click **"Run LLM Assessment & Suggestions"** to get:

- **5-Dimension Rating**: Strategy, Efficiency, Verification, Error Recovery, Completeness — each rated Strong / Adequate / Weak with reasoning
- **Key Findings**: Strengths and weaknesses with supporting evidence
- **Recommendation**: One actionable improvement sentence
- **Actionable Suggestions**: Priority-sorted list of specific improvements with:
  - Category and title
  - Root cause analysis
  - Suggested fix
  - Affected steps
  - Estimated savings

### LLM Comparison (from Compare view)

Click **"Run LLM Comparison"** to get:
- Per-dimension rankings across all selected trajectories
- Comparative summary highlighting key differences
- Recommendation for which approach worked best and why

---

## 8. Tips & FAQ

### How many trajectories do I need?

| Goal | Minimum | Recommended |
|------|---------|-------------|
| View a behavioral profile | 1 | 1 |
| Build a ground truth | 2 (passing) | 3–5 (passing) for a richer GT |
| Run quality assessment | 1 + GT | 1 + GT from 3+ passing traces |
| Compare trajectories | 2 + GT | 3–5 + GT |

### Can I reuse a Ground Truth?

Yes. After building a GT, click **"Export GT"** to download `merged_gt.json`. Next time, use the **"Import merged GT JSON"** option to skip re-uploading passing trajectories.

### Is my data stored permanently?

No. AgentLens uses in-memory storage. **All uploaded data is cleared when the app restarts.** Export anything you want to keep (GT JSON, raw trajectories).

### What if I don't have ATIF data?

AgentLens works with evaluation platform, OpenHands, and raw JSON formats too. You'll get the core analysis (stages, tools, files, coherence, quality assessment, comparison) but won't see Human Experience metrics (HX Score, time breakdowns, latency/token charts).

### What does the HX Score measure?

The HX Score (0–100) is a composite metric of the human experience during an agent session:

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| Autonomy | 30% | How independently the agent worked (fewer human inputs = higher) |
| Low Friction | 30% | How little time was spent waiting for permissions |
| Responsiveness | 25% | How fast the LLM responded (lower latency = higher) |
| Stability | 15% | How stable the context was (fewer compactions = higher) |

### Can I use this for benchmarking agents?

Absolutely. Upload trajectories from different agents solving the same task, build a ground truth from passing runs, and use the Compare view to rank them objectively on quality, efficiency, and human experience.
