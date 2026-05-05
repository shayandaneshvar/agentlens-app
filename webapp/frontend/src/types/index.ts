/* API response types matching the backend schemas. */

export interface TraceInfo {
  trace_id: string;
  label: string;
  format: string;
  passed: boolean | null;
  state_count: number;
  tool_count: number;
  file_count: number;
  model: string;
  task: string;
  benchmark: string;
}

export interface TraceListResponse {
  traces: TraceInfo[];
  passing_count: number;
  has_ground_truth: boolean;
  ground_truth_id: string | null;
}

export interface ProfileResponse {
  trace_id: string;
  state_count: number;
  file_count: number;
  tool_count: number;
  coherence: number;
  coherence_label: string;
  stage_distribution: Record<string, number>;
  stage_percentages: Record<string, number>;
  tool_distribution: Record<string, number>;
  files_touched: string[];
  fingerprint: string;
  fingerprint_detail: string[];
  operation_types: Record<string, number>;
  completed: boolean | null;
  stage_sequence: string[];
  tool_sequence: string[];
  exploration_ratio: number;
  files_modified: number;
  files_read_only: number;
  model: string;
  agent: string;
  task: string;
  benchmark: string;
  human_input_count: number | null;
  subagent_count: number | null;
  active_time_ms: number | null;
  compaction_count: number | null;
  // Human Experience metrics (ATIF only — null for other formats)
  wall_time_ms: number | null;
  permission_wait_ms: number | null;
  human_experience_score: number | null;
  hx_breakdown: Record<string, number> | null;
  time_decomposition: Record<string, number> | null;
  step_latencies: number[] | null;
  step_token_cumulative: number[] | null;
  human_input_positions: number[] | null;
}

export interface GroundTruthResponse {
  gt_id: string;
  source_count: number;
  state_count: number;
}

export interface FailureReason {
  reason: string;
  detail: string;
  severity: string;
}

export interface DivergencePoint {
  step: number;
  description: string;
  expected_next: string;
}

export interface StageCoverageDetail {
  matched: number;
  total: number;
  percent: number;
}

export interface DivergenceSegment {
  start_step: number;
  end_step: number;
  expected_states: Array<{ tool: string; file_path: string; intent_stage: string; resulting_state: string }>;
  candidate_activity: Array<{ tool: string; file_path: string; intent_stage: string }>;
  stage_context: string;
}

export interface StageComparison {
  expected_steps: Array<{ tool: string; file_path: string; resulting_state: string }>;
  matched_steps: Array<{ tool: string; file_path: string; resulting_state: string }>;
  missing_steps: Array<{ tool: string; file_path: string; resulting_state: string }>;
  extra_steps: Array<{ tool: string; file_path: string; resulting_state: string }>;
  ordering_preserved: boolean;
  effort_ratio: number;
}

export interface RetryLoop {
  start_step: number;
  end_step: number;
  tool: string;
  file_path: string;
  count: number;
}

export interface Backtrack {
  step: number;
  from_stage: string;
  to_stage: string;
}

export interface RedundantStep {
  step: number;
  tool: string;
  file_path: string;
}

export interface UnnecessaryExploration {
  step: number;
  tool: string;
  file_path: string;
}

export interface CyclicPattern {
  start_step: number;
  end_step: number;
  pattern_length: number;
  repetitions: number;
  pattern_signature: string[];
}

export interface ToolInefficiency {
  tool: string;
  retries: number;
  backtracks: number;
  cycles: number;
  redundant: number;
  unnecessary: number;
  total_wasted: number;
}

export interface InefficiencyReport {
  retry_loops: RetryLoop[];
  backtracks: Backtrack[];
  redundant_steps: RedundantStep[];
  unnecessary_explorations: UnnecessaryExploration[];
  cyclic_patterns: CyclicPattern[];
  retry_loop_count: number;
  backtrack_count: number;
  redundant_step_count: number;
  unnecessary_exploration_count: number;
  cyclic_pattern_count: number;
  total_wasted_steps: number;
  severity_score: number;
  wasted_input_tokens: number;
  wasted_output_tokens: number;
  total_input_tokens: number;
  total_output_tokens: number;
  per_tool_breakdown: ToolInefficiency[];
}

export interface QualitySignal {
  signal_type: string;
  description: string;
  severity: string;
  evidence: string[];
}

export interface QualityReport {
  verdict: string;
  quality_tier: string;
  quality_score: number;
  failure_reasons: FailureReason[];
  strengths: string[];
  divergence_point: DivergencePoint | null;
  stage_coverage: Record<string, StageCoverageDetail>;
  key_metrics: Record<string, number>;
  divergence_points: DivergenceSegment[];
  stage_comparison: Record<string, StageComparison>;
  inefficiencies: InefficiencyReport | null;
  quality_signals: QualitySignal[];
  stage_order_match: boolean;
}

export interface AssessResponse {
  trace_id: string;
  match_metrics: Record<string, unknown>;
  quality_report: QualityReport;
}

export interface ComparisonStateSummary {
  state_id: string;
  step: number;
  tool: string;
  file_path: string;
  intent_stage: string;
  resulting_state: string;
  content_description: string;
}

export interface AlignmentPair {
  gt_index: number;
  candidate_index: number;
}

export interface ComparisonData {
  gt_path: ComparisonStateSummary[];
  candidate_path: ComparisonStateSummary[];
  alignment: AlignmentPair[];
  gt_matched_indexes: number[];
  candidate_matched_indexes: number[];
  terminal_state_match: boolean;
}

export interface AssessWithGTResponse {
  trace_id: string;
  gt_source_count: number;
  gt_state_count: number;
  match_metrics: Record<string, unknown>;
  quality_report: QualityReport;
  process_coverage: number;
  missing_tools: string[];
  file_coverage: number;
  missing_files: string[];
  comparison: ComparisonData | null;
}

export interface CohortEntry {
  label: string;
  quality_score: number;
  quality_tier: string;
  rank: number;
  top_failure_reason?: string;
}

export interface CohortRanking {
  passing: CohortEntry[];
  failing: CohortEntry[];
  summary: Record<string, unknown>;
}

export interface BatchAssessResponse {
  ranking: CohortRanking;
  trajectories: Array<{
    trace_id: string;
    label: string;
    quality_report: QualityReport;
  }>;
}

// Trace visualization types
export interface TraceState {
  state_id: string;
  step: number;
  tool_used: string | null;
  observation: string;
  resulting_state: string;
  intent_stage: string;
  file_path: string;
  content_description: string;
  metadata: Record<string, unknown>;
}

export interface TraceTransition {
  transition_id: string;
  from_state: string;
  to_state: string;
  action_type: string;
}

export interface TraceVisualization {
  initial_state: string;
  states: Record<string, TraceState>;
  transitions: TraceTransition[];
  branches: Record<string, string[]>;
}

// LLM behavioral assessment types
export interface LLMDimensionScore {
  reasoning: string;
  rating: 'strong' | 'adequate' | 'weak';
}

export interface LLMFinding {
  type: 'strength' | 'weakness';
  observation: string;
  evidence: string;
}

export interface LLMAssessment {
  summary: string;
  dimensions: Record<string, LLMDimensionScore>;
  key_findings: LLMFinding[];
  overall_rating: 'strong' | 'adequate' | 'weak';
  recommendation: string;
}

export interface LLMAssessResponse {
  trace_id: string;
  model_used: string;
  assessment: LLMAssessment;
  quality_score: number;
  verdict: string;
}

export interface LLMSuggestion {
  priority: 'high' | 'medium' | 'low';
  category: string;
  title: string;
  root_cause: string;
  suggestion: string;
  affected_steps: number[];
  estimated_savings: string;
}

export interface LLMSuggestionsResponse {
  trace_id: string;
  model_used: string;
  suggestions: LLMSuggestion[];
  improvement_summary: string;
}

/* ── Comparison types ─────────────────────────────────────────── */

export interface CandidateMetrics {
  quality_score: number;
  verdict: string;
  coverage_percent: number;
  coherence_score: number;
  stage_completeness: number;
  workflow_similarity: number;
  f1_score: number;
  bottleneck_coverage: number;
  bottleneck_stage: string;
  process_coverage: number;
  file_coverage: number;
}

export interface CandidateStageDetail {
  matched: number;
  missing: number;
  extra: number;
  effort_ratio: number;
  ordering_preserved: boolean;
}

export interface CandidateInefficiencies {
  total_wasted_steps: number;
  retry_loop_count: number;
  backtrack_count: number;
  redundant_step_count: number;
  cyclic_pattern_count: number;
  severity_score: number;
  wasted_input_tokens: number;
  wasted_output_tokens: number;
  total_input_tokens: number;
  total_output_tokens: number;
}

export interface CompareCandidate {
  trace_id: string;
  label: string;
  passed: boolean | null;
  model: string;
  agent: string;
  state_count: number;
  metrics: CandidateMetrics;
  inefficiencies: CandidateInefficiencies;
  stage_detail: Record<string, CandidateStageDetail>;
  matched_gt_state_ids: string[];
  strengths: string[];
  failure_reasons: Array<{ severity: string; reason: string; detail: string }>;
  human_input_count: number | null;
  subagent_count: number | null;
  active_time_ms: number | null;
  compaction_count: number | null;
  // Human Experience metrics
  wall_time_ms: number | null;
  permission_wait_ms: number | null;
  human_experience_score: number | null;
  hx_breakdown: Record<string, number> | null;
  time_decomposition: Record<string, number> | null;
  step_latencies: number[] | null;
}

export interface CompareResponse {
  candidates: CompareCandidate[];
  gt: Record<string, unknown>;
  gt_state_ids: string[];
}

export interface LLMDimensionComparison {
  ranking: string[];
  analysis: string;
}

export interface LLMKeyDifference {
  aspect: string;
  observation: string;
  labels_compared: string[];
}

export interface LLMComparativeResult {
  comparative_summary: string;
  dimension_comparison: Record<string, LLMDimensionComparison>;
  key_differences: LLMKeyDifference[];
  recommendation: string;
}

export interface LLMCompareResponse {
  individual: LLMAssessResponse[];
  comparative: LLMComparativeResult;
  model_used: string;
}
