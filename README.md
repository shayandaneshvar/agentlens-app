# AgentLens

AgentLens applies state-machine methodology to analyze coding agent behavior. It extends a proven approach from UX testing agents to code-agent/SWE-bench evaluation. Key capabilities include:

- Collecting agent trajectories that successfully resolve issues
- Constructing a dominator tree from these trajectories to show that it can:
(1) Identify key steps (e.g., read issue → locate code → modify → test)
(2) Detect failure cases where critical steps are skipped
(3) Produce interpretable diagnostic reports

## AgentLens Webapp

AgentLens is also available as a web app for interactive analysis, profiling, and comparison of agent trajectories. See the [AgentLens User Guide](webapp/AGENTLENS_USER_GUIDE.md) for instructions on how to use it.

## AgentLens-Bench Dataset

A benchmark dataset of **1,815 fully-annotated coding agent trajectories** across 47 tasks and 8 frontier models, with quality assessments, waste detection, and ground-truth PTA graphs.

| Metric | Value |
|--------|-------|
| Tasks | 47 |
| Trajectories | 1,815 (1,136 pass / 679 fail) |
| Models | 8 (Claude, GPT, Gemini families) |
| Annotations | 40 columns per trajectory |
| Quality tiers | Ideal (20.2%), Solid (69.1%), Lucky (10.7%) |

See [`agentlens-bench/README.md`](agentlens-bench/README.md) for full documentation, column reference, and usage examples.

## Contributing

This project welcomes contributions and suggestions. Please open an issue or pull request.
