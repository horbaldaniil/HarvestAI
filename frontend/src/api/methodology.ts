import { api } from "./client";

export type AlgorithmFamily = "xgboost" | "rf" | "lstm";
export type CropType = "wheat" | "corn" | "sunflower";

export interface ModelMetricsBlock {
  rmse: number;
  mae: number;
  r2: number;
  n: number;
}

export interface ModelInfo {
  crop: CropType;
  family: AlgorithmFamily;
  version: string;
  metrics: {
    train?: ModelMetricsBlock;
    val?: ModelMetricsBlock;
    test?: ModelMetricsBlock;
  } | null;
  trained_at: string | null;
  feature_count: number;
}

export interface MethodologyOverview {
  data_sources: string[];
  oblast_count: number;
  sample_count_per_oblast: number | null;
  year_range: number[];
  train_val_test_split: string;
  active_algorithm: string;
  models: ModelInfo[];
}

export interface CoverageEntry {
  oblast: string;
  oblast_uk: string | null;
  sample_count: number;
  iso_3166_2: string | null;
}

export interface CoverageResponse {
  oblasts: CoverageEntry[];
  samples_geojson: {
    type: string;
    features: GeoJSON.Feature[];
  };
}

export async function getMethodologyOverview(): Promise<MethodologyOverview> {
  const { data } = await api.get<MethodologyOverview>("/api/methodology");
  return data;
}

export async function getMethodologyMetrics(): Promise<Record<string, unknown>> {
  const { data } = await api.get("/api/methodology/metrics");
  return data;
}

export async function getCoverage(): Promise<CoverageResponse> {
  const { data } = await api.get<CoverageResponse>("/api/methodology/coverage");
  return data;
}
