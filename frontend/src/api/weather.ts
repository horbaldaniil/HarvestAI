import { api } from "./client";
import type { CropType } from "./dashboard";

/**
 * Forecast row used by both the BottomPanel forecast card and the dedicated
 * /weather page. The latter expects the Week 8 fields (wind/cloud/soil) to
 * be populated; the former tolerates them being null.
 */
export interface WeatherDay {
  observed_on: string;
  temp_min_c: number | null;
  temp_max_c: number | null;
  temp_mean_c: number | null;
  precip_mm: number | null;
  humidity_pct: number | null;
  radiation_mj: number | null;
  wind_speed_max_ms: number | null;
  cloud_cover_pct: number | null;
  soil_moisture_0_10cm: number | null;
  is_forecast: boolean;
}

export type AdviceSeverity = "info" | "warning" | "critical";

export interface WeatherAdvice {
  severity: AdviceSeverity;
  title: string;
  detail: string;
}

export interface WeatherDetail {
  field_id: number;
  field_name: string;
  crop_type: CropType;
  centroid_lat: number | null;
  centroid_lon: number | null;
  /** Upcoming N-day forecast (is_forecast = true rows). */
  days: WeatherDay[];
  /**
   * Past N-day actuals (is_forecast = false, observed_on < today).
   * Empty array when no historical data has been collected for this
   * field — the UI hides the "Останні 14 днів" card in that case.
   */
  history_days: WeatherDay[];
  advices: WeatherAdvice[];
}

export async function fetchForecast(
  fieldId: number,
  days = 14,
): Promise<WeatherDay[]> {
  const { data } = await api.get<WeatherDay[]>(
    `/api/fields/${fieldId}/weather`,
    { params: { days, forecast_only: true } },
  );
  return data;
}

export async function getWeatherDetail(
  fieldId: number,
  days = 14,
): Promise<WeatherDetail> {
  const { data } = await api.get<WeatherDetail>(
    `/api/fields/${fieldId}/weather/detail`,
    { params: { days } },
  );
  return data;
}
