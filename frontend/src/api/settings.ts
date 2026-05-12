import { api } from "./client";

export type Currency = "UAH" | "USD" | "EUR";

/**
 * One entry per `CropType` slug. `null` = user hasn't set a price.
 * Order mirrors `ALL_CROPS` in `api/fields.ts` and the
 * `CropPricesRead` Pydantic schema in `backend/app/routers/settings.py`
 * — keep all three in sync when adding a 14th crop.
 */
export interface CropPrices {
  currency: Currency;
  wheat: number | null;
  corn: number | null;
  sunflower: number | null;
  soybean: number | null;
  rapeseed: number | null;
  barley: number | null;
  rye: number | null;
  oats: number | null;
  buckwheat: number | null;
  peas: number | null;
  sugar_beet: number | null;
  potato: number | null;
  corn_silage: number | null;
}

export type CropPricesUpdate = Partial<CropPrices>;

export async function getCropPrices(): Promise<CropPrices> {
  const { data } = await api.get<CropPrices>("/api/settings/crop-prices");
  return data;
}

export async function updateCropPrices(
  payload: CropPricesUpdate,
): Promise<CropPrices> {
  const { data } = await api.put<CropPrices>("/api/settings/crop-prices", payload);
  return data;
}


// ─── Profile ─────────────────────────────────────────────────

/**
 * Flat shape combining the `users.full_name` column with two JSONB
 * fields (`farm_name`, `phone_number`). The backend hides the split
 * behind a single `/api/settings/profile` endpoint.
 */
export interface Profile {
  full_name: string | null;
  farm_name: string | null;
  phone_number: string | null;
}

export type ProfileUpdate = Partial<Profile>;

export async function getProfile(): Promise<Profile> {
  const { data } = await api.get<Profile>("/api/settings/profile");
  return data;
}

export async function updateProfile(payload: ProfileUpdate): Promise<Profile> {
  const { data } = await api.put<Profile>("/api/settings/profile", payload);
  return data;
}


// ─── AI-suggested crop prices ────────────────────────────────

/**
 * One-shot LLM call returning plausible procurement prices for the
 * 13 supported crops in the requested currency. The response shape
 * matches `CropPrices` so the form can populate it directly; values
 * with no AI estimate come back as `null`.
 */
export async function suggestCropPrices(
  currency: Currency = "UAH",
): Promise<CropPrices> {
  const { data } = await api.post<CropPrices>(
    "/api/settings/crop-prices/suggest",
    { currency },
  );
  return data;
}
