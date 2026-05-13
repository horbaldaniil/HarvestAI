import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { Coins, Settings as SettingsIcon } from "lucide-react";

import type { DashboardFieldRow, CropType } from "@/api/dashboard";
import { ALL_CROPS } from "@/api/fields";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useCropPrices } from "@/hooks/useCropPrices";
import { currencyLabelUk } from "@/lib/utils";

interface Props {
  fields: DashboardFieldRow[];
}

/**
 * Income projection = sum over fields of `predicted_tha × area_ha × price_per_ton`,
 * grouped by crop. Empty-state nudges the user to set prices on /settings.
 * No prices = card is hidden? — no, we still show the card with a "set prices"
 * CTA so the feature is discoverable.
 */
export function IncomeProjectionCard({ fields }: Props) {
  const { t } = useTranslation();
  const { data: prices } = useCropPrices();
  const navigate = useNavigate();

  const breakdown = useMemo(() => {
    if (!prices) return null;
    // Initialize the per-crop tallies from `ALL_CROPS` so every crop
    // gets a row regardless of whether the user has fields for it
    // (the render-side filter later drops empties from the display).
    const perCrop = Object.fromEntries(
      ALL_CROPS.map((c) => [c, { yield_t: 0, income: 0 }]),
    ) as Record<CropType, { yield_t: number; income: number }>;
    let total = 0;
    let hasAnyData = false;

    for (const f of fields) {
      if (f.predicted_tha === null) continue;
      const tons = f.predicted_tha * f.area_ha;
      perCrop[f.crop_type].yield_t += tons;
      const price = (prices as Record<CropType, number | null>)[f.crop_type];
      if (price !== null && price !== undefined) {
        const income = tons * price;
        perCrop[f.crop_type].income += income;
        total += income;
        hasAnyData = true;
      }
    }
    return { perCrop, total, hasAnyData };
  }, [fields, prices]);

  const currency = prices?.currency ?? "UAH";
  const pricesSet =
    !!prices &&
    ALL_CROPS.some((c) => {
      const v = prices[c];
      return v !== null && v !== undefined;
    });

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          Очікуваний дохід
        </CardTitle>
        <Coins className="h-5 w-5 text-amber-600" />
      </CardHeader>
      <CardContent>
        {pricesSet && breakdown?.hasAnyData ? (
          <>
            <div className="text-3xl font-bold tabular-nums">
              {formatMoney(breakdown.total)} {currencyLabelUk(currency)}
            </div>
            <div className="mt-2 space-y-1 text-xs text-muted-foreground">
              {(Object.entries(breakdown.perCrop) as [CropType, { yield_t: number; income: number }][])
                .filter(([, v]) => v.yield_t > 0)
                .map(([crop, v]) => (
                  <div key={crop} className="flex justify-between">
                    <span>{t(`fields.crops.${crop}`)}</span>
                    <span className="tabular-nums">
                      {v.yield_t.toFixed(1)} т · {formatMoney(v.income)} {currencyLabelUk(currency)}
                    </span>
                  </div>
                ))}
            </div>
          </>
        ) : (
          <div className="space-y-3">
            <div className="text-sm text-muted-foreground">
              Налаштуйте ціну за тонну для своїх культур — побачите розрахунок очікуваного доходу.
            </div>
            <Button
              size="sm"
              variant="outline"
              onClick={() => navigate("/settings")}
            >
              <SettingsIcon className="h-3.5 w-3.5" />
              Задати ціни
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function formatMoney(v: number): string {
  // Compact: 1 234 567 → "1.2M", 12 345 → "12.3K"; otherwise full.
  if (Math.abs(v) >= 1_000_000) return (v / 1_000_000).toFixed(2) + "M";
  if (Math.abs(v) >= 10_000) return (v / 1_000).toFixed(1) + "K";
  return v.toFixed(0);
}
