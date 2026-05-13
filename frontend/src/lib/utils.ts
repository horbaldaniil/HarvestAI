import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

import type { Currency } from "@/api/settings";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}


/**
 * Ukrainian display label for a 3-letter ISO currency code.
 *
 * Returns the genitive plural ("гривень" / "доларів" / "євро") because
 * every consumer puts a numeric value in front of it:
 *   "2 500 000 гривень" / "12 500 доларів" / "8 200 євро".
 *
 * The ISO code stays the canonical storage key (used by
 * `CropPricesCard` dropdown, `settings_json` backend payload, etc.) —
 * this helper is purely a display concern.
 */
export function currencyLabelUk(code: Currency): string {
  switch (code) {
    case "UAH":
      return "гривень";
    case "USD":
      return "доларів";
    case "EUR":
      return "євро";
  }
}
