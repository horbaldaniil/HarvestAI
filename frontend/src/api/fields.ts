import type { Polygon as GJPolygon, Point as GJPoint } from "geojson";
import { api } from "./client";

export type CropType = "wheat" | "corn" | "sunflower";

export interface FieldRead {
  id: number;
  name: string;
  crop_type: CropType;
  season_year: number;
  geometry: GJPolygon;
  centroid: GJPoint;
  area_ha: number;
  color: string;
  created_at: string;
  updated_at: string;
}

export interface FieldCreatePayload {
  name: string;
  crop_type: CropType;
  season_year: number;
  geometry: GJPolygon;
  color?: string | null;
}

export interface FieldUpdatePayload {
  name?: string;
  crop_type?: CropType;
  season_year?: number;
  geometry?: GJPolygon;
  color?: string | null;
}

export async function listFields(): Promise<FieldRead[]> {
  const { data } = await api.get<FieldRead[]>("/api/fields");
  return data;
}

export async function getField(id: number): Promise<FieldRead> {
  const { data } = await api.get<FieldRead>(`/api/fields/${id}`);
  return data;
}

export async function createField(payload: FieldCreatePayload): Promise<FieldRead> {
  const { data } = await api.post<FieldRead>("/api/fields", payload);
  return data;
}

export async function updateField(
  id: number,
  payload: FieldUpdatePayload,
): Promise<FieldRead> {
  const { data } = await api.patch<FieldRead>(`/api/fields/${id}`, payload);
  return data;
}

export async function deleteField(id: number): Promise<void> {
  await api.delete(`/api/fields/${id}`);
}
