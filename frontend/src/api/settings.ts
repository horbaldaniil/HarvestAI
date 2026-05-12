import { api } from "./client";

export type Currency = "UAH" | "USD" | "EUR";

export interface CropPrices {
  currency: Currency;
  wheat: number | null;
  corn: number | null;
  sunflower: number | null;
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
