import type { CropType } from "@/api/fields";

export const CROP_COLORS: Record<CropType, string> = {
  wheat: "#f0c419",
  corn: "#e67e22",
  sunflower: "#f1c40f",
};

export function cropColor(crop: CropType, override?: string | null): string {
  return override || CROP_COLORS[crop];
}
