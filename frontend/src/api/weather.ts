import { api } from "./client";

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
