"""SWE Trace SDK — analyse and compare coding-agent execution traces.

Quick-start
-----------
>>> from swe_trace_sdk import trace, match, intent
>>> candidate = trace.load("path/to/chat-export-logs.json")
>>> ground_truth = trace.merge([trace1, trace2])
>>> result = match.run(candidate, ground_truth)
>>> print(result.metrics.coverage_percent)

Quality assessment
------------------
>>> report = match.quality_assessment(result, candidate, ground_truth)
>>> print(report.verdict, report.quality_tier, report.quality_score)

Cohort ranking
--------------
>>> ranking = match.rank_in_cohort([("run1", r1), ("run2", r2)])
>>> print(ranking.passing, ranking.failing)

See the project README for full usage details.
"""

__version__ = "0.2.0"
