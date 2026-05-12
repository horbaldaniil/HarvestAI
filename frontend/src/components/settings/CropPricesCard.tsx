import { useEffect, useState } from "react";
import { Loader2, Save } from "lucide-react";

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
import { useCropPrices, useUpdateCropPrices } from "@/hooks/useCropPrices";
import type { Currency } from "@/api/settings";

/**
 * Per-crop price editor. Values feed the dashboard's Income projection card
 * and the PDF portfolio report (when set). Empty input = "not set" — we
 * send null to clear stored prices on submit.
 */
export function CropPricesCard() {
  const { data, isLoading } = useCropPrices();
  const update = useUpdateCropPrices();

  const [wheat, setWheat] = useState<string>("");
  const [corn, setCorn] = useState<string>("");
  const [sunflower, setSunflower] = useState<string>("");
  const [currency, setCurrency] = useState<Currency>("UAH");

  useEffect(() => {
    if (!data) return;
    setWheat(data.wheat !== null && data.wheat !== undefined ? String(data.wheat) : "");
    setCorn(data.corn !== null && data.corn !== undefined ? String(data.corn) : "");
    setSunflower(data.sunflower !== null && data.sunflower !== undefined ? String(data.sunflower) : "");
    setCurrency(data.currency ?? "UAH");
  }, [data]);

  const parse = (s: string): number | null => {
    const v = s.trim();
    if (!v) return null;
    const n = Number(v);
    return Number.isFinite(n) && n >= 0 ? n : null;
  };

  const handleSave = () => {
    update.mutate({
      currency,
      wheat: parse(wheat),
      corn: parse(corn),
      sunflower: parse(sunflower),
    });
  };

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
            <PriceField label="Пшениця" value={wheat} onChange={setWheat} currency={currency} />
            <PriceField label="Кукурудза" value={corn} onChange={setCorn} currency={currency} />
            <PriceField label="Соняшник" value={sunflower} onChange={setSunflower} currency={currency} />
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
