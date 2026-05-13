import { api } from "./client";

export type AlgorithmFamily =
  | "xgboost"
  | "rf"
  | "lstm"
  | "lightgbm"
  | "stack";

/**
 * 13 supported crops (Phase 4 expansion). The string values mirror
 * `backend/app/db/models/enums.py:CropType` — DO NOT rename without a
 * coordinated migration of the parquet / CSV / i18n bundle.
 */
export type CropType =
  | "wheat"
  | "corn"
  | "sunflower"
  | "soybean"
  | "rapeseed"
  | "barley"
  | "rye"
  | "oats"
  | "buckwheat"
  | "peas"
  | "sugar_beet"
  | "potato"
  | "corn_silage";

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

// ─── Phase 4 v3 endpoints — scientific metric panel ───────────

export interface LeaderboardRow {
  crop: CropType;
  family: AlgorithmFamily;
  n_test: number | null;
  n_train: number | null;
  test_r2: number | null;
  test_rmse: number | null;
  test_mae: number | null;
  test_mape: number | null;
  loocv_r2: number | null;
  loocv_rmse: number | null;
  kfold_r2_mean: number | null;
  kfold_r2_std: number | null;
  pinball_q05: number | null;
  pinball_q95: number | null;
  interval_coverage_90pct: number | null;
}

export interface LeaderboardResponse {
  rows: LeaderboardRow[];
  metadata?: Record<string, unknown>;
  note?: string;
}

export interface PerOblastResidual {
  iso_3166_2: string;
  oblast: string;
  mean_abs_residual: number;
  max_abs_residual: number;
  n: number;
}

export interface PredVsActualPoint {
  actual: number;
  predicted: number;
  oblast: string;
  iso_3166_2: string;
}

export interface LearningCurvePoint {
  n_train: number;
  fraction: number;
  test_r2: number | null;
  test_mae: number | null;
  test_rmse: number | null;
}

export interface EvalSplitMetrics {
  n: number;
  rmse: number | null;
  mae: number | null;
  r2: number | null;
  mape: number | null;
  smape: number | null;
}

export interface EvalFamilyBody {
  skipped?: boolean;
  reason?: string;
  n_train?: number;
  n_test?: number;
  test?: EvalSplitMetrics;
  loocv_oblast?: EvalSplitMetrics;
  repeated_kfold_5x3?: {
    n_folds: number;
    r2_mean: number | null;
    r2_std: number | null;
    r2_min: number | null;
    r2_max: number | null;
  };
  pinball_loss_q05?: number;
  pinball_loss_q95?: number;
  interval_coverage_90pct?: number;
  per_oblast_residuals?: PerOblastResidual[];
  pred_vs_actual?: PredVsActualPoint[];
  learning_curve?: LearningCurvePoint[];
  global_shap?: Record<string, number> | null;
  permutation_importance?: Record<string, number>;
  /**
   * Phase-4 1-D Partial Dependence (Friedman 2001). One entry per
   * top-3 feature picked by permutation importance. Stack family
   * returns `{}` because PDP isn't meaningful on the meta-input space.
   */
  partial_dependence?: Record<string, { grid: number[]; pdp: number[] }>;
}

export interface EvalCropBody {
  [family: string]: EvalFamilyBody;
}

export interface EvaluationV3Response {
  metadata: {
    evaluated_at?: string;
    training_set?: string;
    features_origin?: string;
    feature_names?: string[];
    splits?: Record<string, string>;
    families?: string[];
    skipped?: Record<string, boolean>;
  };
  crops: Record<string, EvalCropBody>;
  note?: string;
}

export async function getLeaderboard(): Promise<LeaderboardResponse> {
  const { data } = await api.get<LeaderboardResponse>(
    "/api/methodology/leaderboard"
  );
  return data;
}

export async function getEvaluationV3(): Promise<EvaluationV3Response> {
  const { data } = await api.get<EvaluationV3Response>(
    "/api/methodology/evaluation_v3"
  );
  return data;
}

// ─── Phase-5 PDF methodology report ──────────────────────────────

/**
 * Download the methodology PDF report as a Blob. Caller is responsible
 * for triggering the browser-side save (see `downloadMethodologyReport`
 * helper below). We use `responseType: 'blob'` so axios doesn't try
 * to parse the PDF bytes as JSON/text.
 */
export async function getMethodologyReportPdf(
  crop: string = "wheat",
  family: string = "stack",
): Promise<Blob> {
  const { data } = await api.get<Blob>("/api/reports/methodology", {
    params: { crop, family },
    responseType: "blob",
  });
  return data;
}

/**
 * Browser-side save helper: fetches the PDF and triggers an automatic
 * download via an <a download> element. Doesn't return until the
 * download is initiated.
 */
export async function downloadMethodologyReport(
  crop: string = "wheat",
  family: string = "stack",
): Promise<void> {
  const blob = await getMethodologyReportPdf(crop, family);
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  const stamp = new Date().toISOString().slice(0, 10);
  a.download = `HarvestAI_methodology_${crop}_${family}_${stamp}.pdf`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Release the object URL after the click event has propagated.
  setTimeout(() => window.URL.revokeObjectURL(url), 1000);
}
