/* Typed fetch wrappers for the backend API. */

import type {
  TraceInfo,
  TraceListResponse,
  ProfileResponse,
  GroundTruthResponse,
  AssessResponse,
  AssessWithGTResponse,
  BatchAssessResponse,
  TraceVisualization,
  LLMAssessResponse,
  LLMSuggestionsResponse,
  CompareResponse,
  LLMCompareResponse,
} from '../types';

const BASE = '/api';

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export async function uploadFile(file: File): Promise<TraceInfo[]> {
  const form = new FormData();
  form.append('file', file);
  return request<TraceInfo[]>(`${BASE}/upload`, { method: 'POST', body: form });
}

export async function uploadBatch(files: File[]): Promise<TraceInfo[]> {
  const form = new FormData();
  for (const f of files) {
    form.append('files', f);
  }
  return request<TraceInfo[]>(`${BASE}/upload-batch`, { method: 'POST', body: form });
}

export async function listTraces(): Promise<TraceListResponse> {
  return request<TraceListResponse>(`${BASE}/traces`);
}

export async function updateTrace(
  id: string,
  patch: { label?: string; passed?: boolean | null },
): Promise<TraceInfo> {
  return request<TraceInfo>(`${BASE}/traces/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  });
}

export async function deleteTrace(id: string): Promise<void> {
  await request<unknown>(`${BASE}/traces/${id}`, { method: 'DELETE' });
}

export async function getProfile(id: string): Promise<ProfileResponse> {
  return request<ProfileResponse>(`${BASE}/traces/${id}/profile`);
}

export async function getVisualization(id: string): Promise<TraceVisualization> {
  return request<TraceVisualization>(`${BASE}/traces/${id}/visualization`);
}

export async function mergeTraces(traceIds: string[]): Promise<GroundTruthResponse> {
  return request<GroundTruthResponse>(`${BASE}/merge`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ trace_ids: traceIds }),
  });
}

export async function assessTrace(id: string): Promise<AssessResponse> {
  return request<AssessResponse>(`${BASE}/assess/${id}`, { method: 'POST' });
}

export async function assessWithGT(id: string, gtFiles: File[]): Promise<AssessWithGTResponse> {
  const form = new FormData();
  for (const f of gtFiles) {
    form.append('files', f);
  }
  return request<AssessWithGTResponse>(`${BASE}/assess-with-gt/${id}`, {
    method: 'POST',
    body: form,
  });
}

export async function exportGT(): Promise<Blob> {
  const res = await fetch(`${BASE}/gt/export`);
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return res.blob();
}

export async function importGT(file: File): Promise<GroundTruthResponse> {
  const form = new FormData();
  form.append('file', file);
  return request<GroundTruthResponse>(`${BASE}/gt/import`, { method: 'POST', body: form });
}

export async function assessWithImportedGT(id: string, gtFile: File): Promise<AssessWithGTResponse> {
  const form = new FormData();
  form.append('file', gtFile);
  return request<AssessWithGTResponse>(`${BASE}/assess-with-imported-gt/${id}`, {
    method: 'POST',
    body: form,
  });
}

export async function assessBatch(): Promise<BatchAssessResponse> {
  return request<BatchAssessResponse>(`${BASE}/assess/batch`, { method: 'POST' });
}

export async function llmAssess(traceId: string): Promise<LLMAssessResponse> {
  return request<LLMAssessResponse>(`${BASE}/traces/${traceId}/llm-assess`, {
    method: 'POST',
  });
}

export async function llmSuggestions(traceId: string): Promise<LLMSuggestionsResponse> {
  return request<LLMSuggestionsResponse>(`${BASE}/traces/${traceId}/llm-suggestions`, {
    method: 'POST',
  });
}

export async function compareTraces(traceIds: string[], gtStrategy: string = 'best_match'): Promise<CompareResponse> {
  return request<CompareResponse>(`${BASE}/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ trace_ids: traceIds, gt_strategy: gtStrategy }),
  });
}

export async function llmCompare(traceIds: string[]): Promise<LLMCompareResponse> {
  return request<LLMCompareResponse>(`${BASE}/compare/llm`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ trace_ids: traceIds }),
  });
}
