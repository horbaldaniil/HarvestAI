import { api } from "./client";

export interface ShapBar {
  name: string;
  value: number | null;
  contribution: number;
}

export interface PredictionRead {
  id: number;
  field_id: number;
  model_name: string;
  model_version: string;
  value_tha: number;
  confidence: number | null;
  shap_top_json: ShapBar[];
  predicted_at: string;
}

export interface JobHandle {
  job_id: string;
  queue: string;
  status_url: string;
}

export async function getLatestPrediction(
  fieldId: number,
): Promise<PredictionRead | null> {
  const { data } = await api.get<PredictionRead | null>(
    `/api/fields/${fieldId}/predictions/latest`,
  );
  return data;
}

export async function recomputePrediction(
  fieldId: number,
): Promise<JobHandle> {
  const { data } = await api.post<JobHandle>(
    `/api/fields/${fieldId}/predictions/recompute`,
  );
  return data;
}
