import type { Polygon as GJPolygon, Point as GJPoint } from "geojson";
import { api } from "./client";

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

/**
 * Canonical iteration order for the 13 supported crops. Mirrors the
 * member order of `CropType` in `backend/app/db/models/enums.py` and
 * drives every crop selector, filter chip and chart axis in the UI.
 *
 * Order: original three (wheat/corn/sunflower) — oilseeds — cereals —
 * legume — root crops — forage. Display groupings match the backend
 * enum's comment blocks so visual reading order is the same on both
 * sides of the wire.
 */
export const ALL_CROPS: CropType[] = [
  "wheat",
  "corn",
  "sunflower",
  "soybean",
  "rapeseed",
  "barley",
  "rye",
  "oats",
  "buckwheat",
  "peas",
  "sugar_beet",
  "potato",
  "corn_silage",
];

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
  /**
   * Set by create / update-with-geometry responses when an auto-fetch was
   * enqueued. The frontend should subscribe to SSE on this job to drive the
   * loading UI and refresh observations when it completes.
   */
  pending_job_id?: string | null;
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
