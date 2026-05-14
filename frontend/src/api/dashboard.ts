import type { Polygon as GJPolygon } from "geojson";

import { api } from "./client";

// Single source of truth lives in `api/fields.ts`. Imported here for
// use in the interface declarations below and re-exported so the
// historical `import { CropType } from "@/api/dashboard"` call sites
// keep working without two parallel declarations drifting apart.
import type { CropType } from "./fields";
export type { CropType };

export interface DashboardKpis {
  total_fields: number;
  total_area_ha: number;
  predicted_total_yield_t: number | null;
  avg_ndvi_current: number | null;
  avg_ndwi_current: number | null;
  active_alerts_count: number;
}

export interface DashboardFieldRow {
  field_id: number;
  name: string;
  crop_type: CropType;
  area_ha: number;
  current_ndvi: number | null;
  current_ndwi: number | null;
  predicted_tha: number | null;
  has_alerts: boolean;
  risk_score: number;
  risk_factors: string[];
  oblast_name: string | null;
  /** Legacy NDVI baseline — kept for back-compat; user UI now uses yield. */
  oblast_avg_ndvi: number | null;
  oblast_baseline_year: number | null;
  /**
   * Mean Держстат yield (t/ha) for this field's (oblast, crop). Populated
   * from `training_set_v3.parquet` real-yield rows. `null` when no
   * published row exists for the combo.
   */
  oblast_avg_yield_tha: number | null;
  /** Year the yield baseline came from (typically 2021). */
  oblast_avg_yield_year: number | null;
  /**
   * Headline test R² for this crop from the v7-hybrid evaluator (test
   * year 2021, real Держстат yields). Frontend uses it to size the
   * OblastComparisonTable "within noise" grey band — weak models get
   * wider tolerance before |Δ%| triggers a red flag.
   */
  model_r2_for_crop: number | null;
  geometry: GJPolygon | null;
}

export interface YearOverYear {
  current_year: number;
  current_year_avg_ndvi: number | null;
  prev_year_avg_ndvi: number | null;
  diff_pct: number | null;
}

export interface FieldYoYDelta {
  field_id: number;
  name: string;
  crop_type: CropType;
  current_year_ndvi: number;
  prev_year_ndvi: number;
  diff_pct: number;
}

export interface CropBreakdownItem {
  crop_type: CropType;
  field_count: number;
  area_ha: number;
}

export interface WeatherDay {
  observed_on: string;
  temp_min_c: number | null;
  temp_max_c: number | null;
  temp_mean_c: number | null;
  precip_mm: number | null;
  humidity_pct: number | null;
  radiation_mj: number | null;
  is_forecast: boolean;
}

export interface FieldWeather {
  field_id: number;
  field_name: string;
  crop_type: CropType;
  centroid_lat: number | null;
  centroid_lon: number | null;
  days: WeatherDay[];
  temp_max_7d: number | null;
  temp_min_7d: number | null;
  temp_avg_7d: number | null;
  precip_sum_7d: number | null;
  heat_stress_days_7d: number;
}

export interface DashboardResponse {
  kpis: DashboardKpis;
  fields: DashboardFieldRow[];
  yoy: YearOverYear;
  top_movers: FieldYoYDelta[];
  crops_breakdown: CropBreakdownItem[];
  weather_by_field: FieldWeather[];
  applied_crops: CropType[];
}

export interface DashboardFilters {
  crops?: CropType[];
}

export async function getDashboard(
  filters: DashboardFilters = {},
): Promise<DashboardResponse> {
  const params: Record<string, string | number> = {};
  if (filters.crops && filters.crops.length > 0) {
    params.crops = filters.crops.join(",");
  }
  const { data } = await api.get<DashboardResponse>("/api/dashboard", { params });
  return data;
}
