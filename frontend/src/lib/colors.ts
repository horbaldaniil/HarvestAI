import type { CropType } from "@/api/fields";

/**
 * Hex palette per crop — must match `CropType.default_color` in
 * `backend/app/db/models/enums.py` so the dashboard donut, methodology
 * charts and field-card thumbnails agree across surfaces. Hues picked
 * to roughly match each crop's visual identity in agricultural
 * photography (wheat=amber, sunflower=gold, peas=fresh green, etc.).
 */
export const CROP_COLORS: Record<CropType, string> = {
  wheat: "#f0c419",        // amber — ripe wheat
  corn: "#e67e22",         // orange — corn cob
  sunflower: "#f1c40f",    // gold — sunflower disk
  soybean: "#8bc34a",      // light green — soybean canopy
  rapeseed: "#fdd835",     // bright yellow — rapeseed flowers
  barley: "#d4a373",       // tan — mature barley
  rye: "#9c8866",          // darker tan — rye
  oats: "#c7b07a",         // warm grey — oat panicles
  buckwheat: "#b08fb8",    // pinkish-purple — buckwheat flowers
  peas: "#7cb342",         // fresh green — pea pods
  sugar_beet: "#5d4037",   // brown — soil/beet
  potato: "#a1887f",       // earthy brown — potato skin
  corn_silage: "#558b2f",  // dark green — whole-plant corn
};

export function cropColor(crop: CropType, override?: string | null): string {
  return override || CROP_COLORS[crop];
}
