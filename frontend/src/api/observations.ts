import { api } from "./client";

export type IndexName = "ndvi" | "evi" | "ndwi" | "savi";

export interface ObservationRead {
  id: number;
  observed_on: string; // ISO date
  ndvi_mean: number | null;
  ndvi_min: number | null;
  ndvi_max: number | null;
  ndvi_std: number | null;
  evi_mean: number | null;
  ndwi_mean: number | null;
  savi_mean: number | null;
  cloud_cover: number | null;
  raster_uri: string | null;
}

export interface JobHandleResponse {
  job_id: string;
  queue: string;
  status_url: string;
}

export interface ListParams {
  since?: string;
  until?: string;
}

export async function listObservations(
  fieldId: number,
  params: ListParams = {},
): Promise<ObservationRead[]> {
  const { data } = await api.get<ObservationRead[]>(
    `/api/fields/${fieldId}/observations`,
    { params },
  );
  return data;
}

export async function refreshObservations(
  fieldId: number,
  yearsBack = 2,
): Promise<JobHandleResponse> {
  const { data } = await api.post<JobHandleResponse>(
    `/api/fields/${fieldId}/observations/refresh`,
    null,
    { params: { years_back: yearsBack } },
  );
  return data;
}

export async function requestHeatmap(
  fieldId: number,
  dateIso: string,
  index: IndexName,
): Promise<JobHandleResponse> {
  const { data } = await api.post<JobHandleResponse>(
    `/api/fields/${fieldId}/observations/${dateIso}/heatmap`,
    null,
    { params: { index } },
  );
  return data;
}

export function heatmapUrl(fieldId: number, dateIso: string, index: IndexName): string {
  const base = import.meta.env.VITE_API_BASE_URL || "";
  return `${base}/api/fields/${fieldId}/heatmaps/${dateIso}_${index}.png`;
}
