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
