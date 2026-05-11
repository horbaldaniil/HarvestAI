import { api } from "./client";

export interface DashboardKpis {
  total_fields: number;
  total_area_ha: number;
  predicted_total_yield_t: number | null;
  avg_ndvi_current: number | null;
  active_alerts_count: number;
}

export interface DashboardFieldRow {
  field_id: number;
  name: string;
  crop_type: "wheat" | "corn" | "sunflower";
  area_ha: number;
  current_ndvi: number | null;
  predicted_tha: number | null;
  has_alerts: boolean;
}

export interface YearOverYear {
  current_year: number;
  current_year_avg_ndvi: number | null;
  prev_year_avg_ndvi: number | null;
  diff_pct: number | null;
}

export interface DashboardResponse {
  kpis: DashboardKpis;
  fields: DashboardFieldRow[];
  yoy: YearOverYear;
}

export async function getDashboard(): Promise<DashboardResponse> {
  const { data } = await api.get<DashboardResponse>("/api/dashboard");
  return data;
}
