import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { Loader2, Save, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  useCropPrices,
  useSuggestCropPrices,
  useUpdateCropPrices,
} from "@/hooks/useCropPrices";
import type { CropPrices, CropPricesUpdate, Currency } from "@/api/settings";
import { ALL_CROPS, type CropType } from "@/api/fields";

/**
 * Per-crop price editor. Values feed the dashboard's Income projection
 * card and the PDF portfolio report (when set). Empty input = "not set"
 * — we send `null` to clear stored prices on submit.
 *
 * Driven by `ALL_CROPS` so adding a new crop (i.e. adding a 14th slug
 * to the enum + `CropPrices` interface) automatically adds a row here
 * without code changes — just a translation key under `fields.crops`.
 */
export function CropPricesCard() {
  const { t } = useTranslation();
  const { data, isLoading } = useCropPrices();
  const update = useUpdateCropPrices();
  const suggest = useSuggestCropPrices();

  // Form state is one string-per-crop record so a 14th crop "just shows
  // up" instead of requiring a new useState pair. Strings (not numbers)
  // because the input is text-typed and "" is a valid "clear" signal.
  const [values, setValues] = useState<Record<CropType, string>>(
    () => Object.fromEntries(ALL_CROPS.map((c) => [c, ""])) as Record<CropType, string>,
  );
  const [currency, setCurrency] = useState<Currency>("UAH");

  useEffect(() => {
    if (!data) return;
    const next = {} as Record<CropType, string>;
    for (const crop of ALL_CROPS) {
      const v = data[crop];
      next[crop] = v !== null && v !== undefined ? String(v) : "";
    }
    setValues(next);
    setCurrency(data.currency ?? "UAH");
  }, [data]);

  const parse = (s: string): number | null => {
    const v = s.trim();
    if (!v) return null;
    const n = Number(v);
    return Number.isFinite(n) && n >= 0 ? n : null;
  };

  const handleSave = () => {
    const payload: CropPricesUpdate = { currency };
    for (const crop of ALL_CROPS) {
      payload[crop] = parse(values[crop]);
    }
    update.mutate(payload);
  };

  /**
   * Populate the form with AI-suggested numbers. We DO NOT save them
   * to the backend — the user reviews, edits, then clicks "Зберегти".
   * Existing user-entered values get overwritten only for keys the AI
   * returned (rest stay). That way "suggest" doesn't wipe a user's
   * carefully-tuned price for one crop they already know.
   */
  const handleSuggest = () => {
    suggest.mutate(currency, {
      onSuccess: (suggested: CropPrices) => {
        setValues((prev) => {
          const next = { ...prev };
          for (const crop of ALL_CROPS) {
            const v = suggested[crop];
            if (v !== null && v !== undefined) {
              next[crop] = String(v);
            }
          }
          return next;
        });
      },
    });
  };

  const setOne = (crop: CropType) => (v: string) =>
    setValues((prev) => ({ ...prev, [crop]: v }));

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Ціни на культури</CardTitle>
        <p className="text-xs text-muted-foreground">
          Ціна за тонну. Використовується для оцінки очікуваного доходу
          на дашборді.
        </p>
      </CardHeader>
      <CardContent className="space-y-3">
        {isLoading ? (
          <div className="flex justify-center py-4">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
          </div>
        ) : (
          <>
            <div>
              <Label>Валюта</Label>
              <Select value={currency} onValueChange={(v) => setCurrency(v as Currency)}>
                <SelectTrigger className="mt-1.5">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="UAH">UAH (грн)</SelectItem>
                  <SelectItem value="USD">USD ($)</SelectItem>
                  <SelectItem value="EUR">EUR (€)</SelectItem>
                </SelectContent>
              </Select>
            </div>
            {/*
              AI suggestion — fills the form (does NOT save) with
              plausible current-season prices for all 13 crops in the
              chosen currency. User reviews each row + clicks Зберегти
              to persist. Existing user-entered values are preserved
              for any crops the AI didn't return.
            */}
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handleSuggest}
              disabled={suggest.isPending || update.isPending}
              className="w-full"
            >
              {suggest.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Sparkles className="h-4 w-4 text-amber-500" />
              )}
              Запропонувати ціни через AI
            </Button>
            {ALL_CROPS.map((crop) => (
              <PriceField
                key={crop}
                label={t(`fields.crops.${crop}`)}
                value={values[crop]}
                onChange={setOne(crop)}
                currency={currency}
              />
            ))}
            <Button
              onClick={handleSave}
              disabled={update.isPending}
              className="w-full"
            >
              {update.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Save className="h-4 w-4" />
              )}
              Зберегти
            </Button>
          </>
        )}
      </CardContent>
    </Card>
  );
}

function PriceField({
  label,
  value,
  onChange,
  currency,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  currency: Currency;
}) {
  return (
    <div>
      <Label>{label}</Label>
      <div className="mt-1.5 flex gap-2">
        <Input
          type="number"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Не задано"
          min={0}
          step={50}
        />
        <span className="flex items-center text-xs text-muted-foreground">
          {currency}/т
        </span>
      </div>
    </div>
  );
}
