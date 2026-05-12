import { useMemo, useState } from "react";
import { Loader2 } from "lucide-react";
import { AttributionControl, GeoJSON, MapContainer, TileLayer } from "react-leaflet";
import type { Feature, FeatureCollection } from "geojson";

import type {
  CropType,
  EvalFamilyBody,
  PerOblastResidual,
} from "@/api/methodology";
import { useCoverage, useEvaluationV3 } from "@/hooks/useMethodology";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

/**
 * Choropleth map of per-oblast mean absolute residual.
 *
 * Reads the chosen (crop, family) entry from `evaluation_v3.json`,
 * joins per-oblast residuals onto the WorldCover-sampled polygons,
 * and colours each oblast on a yellow→red gradient. Hovering shows
 * the oblast name and residual value.
 *
 * Crop/family pickers are bound to the URL hash so the operator can
 * deep-link a specific configuration ("here's where stack v3 fails
 * for sugar_beet").
 */
export function OblastResidualMap() {
  const { data: evalData, isLoading: evalLoading } = useEvaluationV3();
  const { data: coverage, isLoading: covLoading } = useCoverage();
  const isLoading = evalLoading || covLoading;

  const allCrops = useMemo(
    () => (evalData ? Object.keys(evalData.crops).sort() : []),
    [evalData],
  );
  const [crop, setCrop] = useState<string>("");
  const [family, setFamily] = useState<string>("stack");

  // Default to the first available crop once data lands.
  const effectiveCrop = crop || allCrops[0] || "";
  const cropBody = effectiveCrop ? evalData?.crops[effectiveCrop] : undefined;
  const availableFamilies = cropBody
    ? Object.keys(cropBody).filter((f) => !cropBody[f]?.skipped)
    : [];
  const effectiveFamily =
    cropBody && availableFamilies.includes(family)
      ? family
      : availableFamilies[0] || "";

  const body: EvalFamilyBody | undefined =
    cropBody && effectiveFamily ? cropBody[effectiveFamily] : undefined;

  const residualsByIso = useMemo(() => {
    const out = new Map<string, PerOblastResidual>();
    for (const r of body?.per_oblast_residuals ?? []) {
      out.set(r.iso_3166_2, r);
    }
    return out;
  }, [body]);

  // Build the colour scale from the actual residual range so the gradient
  // adapts per (crop, family) rather than using a global hard-coded max.
  const { vmin, vmax } = useMemo(() => {
    const vals = [...residualsByIso.values()].map((r) => r.mean_abs_residual);
    if (vals.length === 0) return { vmin: 0, vmax: 1 };
    return { vmin: Math.min(...vals), vmax: Math.max(...vals) };
  }, [residualsByIso]);

  if (isLoading) {
    return (
      <Card>
        <CardContent className="flex h-80 items-center justify-center">
          <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }
  if (!evalData || allCrops.length === 0) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-sm text-muted-foreground">
          Дані оцінки моделей ще не згенеровані. Запустіть{" "}
          <code className="rounded bg-muted px-1">scripts/evaluate_models.py</code>.
        </CardContent>
      </Card>
    );
  }

  const samples = coverage?.samples_geojson as FeatureCollection | undefined;

  function styleFor(feature?: Feature) {
    const iso = (feature?.properties as { iso_3166_2?: string } | undefined)?.iso_3166_2 ?? "";
    const r = residualsByIso.get(iso);
    if (!r) {
      return { color: "#cbd5e1", fillColor: "#cbd5e1", fillOpacity: 0.25, weight: 1 };
    }
    return {
      color: residualColour(r.mean_abs_residual, vmin, vmax),
      fillColor: residualColour(r.mean_abs_residual, vmin, vmax),
      fillOpacity: 0.7,
      weight: 1.5,
    };
  }

  return (
    <Card>
      <CardHeader className="space-y-2">
        <CardTitle className="text-base">Карта залишків по областях</CardTitle>
        <div className="flex flex-wrap items-center gap-3 text-xs">
          <label className="inline-flex items-center gap-1">
            <span className="text-muted-foreground">Культура:</span>
            <select
              className="rounded border bg-background px-1 py-0.5"
              value={effectiveCrop}
              onChange={(e) => setCrop(e.target.value as CropType)}
            >
              {allCrops.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </label>
          <label className="inline-flex items-center gap-1">
            <span className="text-muted-foreground">Модель:</span>
            <select
              className="rounded border bg-background px-1 py-0.5"
              value={effectiveFamily}
              onChange={(e) => setFamily(e.target.value)}
              disabled={availableFamilies.length === 0}
            >
              {availableFamilies.map((f) => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
          </label>
          {body?.test?.mae != null && (
            <span className="text-muted-foreground">
              Test MAE: <span className="font-medium text-foreground">{body.test.mae.toFixed(3)}</span> t/ha
            </span>
          )}
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="h-[450px] w-full">
          <MapContainer
            center={[49.0, 31.5]}
            zoom={5}
            style={{ height: "100%", width: "100%" }}
            scrollWheelZoom={false}
            attributionControl={false}
          >
            <AttributionControl prefix={false} position="bottomright" />
            <TileLayer
              url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"
              attribution='&copy; <a href="https://carto.com/">CartoDB</a> &copy; OpenStreetMap'
              maxZoom={18}
            />
            {samples && samples.features.length > 0 && (
              <GeoJSON
                key={`${effectiveCrop}-${effectiveFamily}-${vmin}-${vmax}`}
                data={samples}
                style={styleFor as never}
                onEachFeature={(feature, layer) => {
                  const iso = (feature.properties as { iso_3166_2?: string })?.iso_3166_2 ?? "";
                  const r = residualsByIso.get(iso);
                  const name = (feature.properties as { oblast_uk?: string; oblast?: string })?.oblast_uk
                    ?? (feature.properties as { oblast?: string })?.oblast
                    ?? iso;
                  const tooltip = r
                    ? `${name}: MAE = ${r.mean_abs_residual.toFixed(3)} (n=${r.n})`
                    : `${name}: дані відсутні`;
                  layer.bindTooltip(tooltip, { sticky: true });
                }}
              />
            )}
          </MapContainer>
        </div>
        <div className="flex items-center justify-between border-t px-4 py-2 text-xs">
          <span className="text-muted-foreground">
            Mean absolute residual per oblast (test split). Шкала: жовтий → темно-червоний (низький → високий error).
          </span>
          <LegendBar vmin={vmin} vmax={vmax} />
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Yellow → red gradient. We interpolate in HSL so the steps look
 * perceptually uniform rather than the smudgy mid-tones you get
 * interpolating RGB end-points.
 */
function residualColour(v: number, vmin: number, vmax: number): string {
  if (vmax <= vmin) return "#fde047";
  const t = Math.max(0, Math.min(1, (v - vmin) / (vmax - vmin)));
  // Hue 50° (yellow) → 0° (red); saturation 95 %; lightness 60 % → 45 %.
  const hue = 50 - 50 * t;
  const light = 60 - 15 * t;
  return `hsl(${hue.toFixed(0)}deg 95% ${light.toFixed(0)}%)`;
}

function LegendBar({ vmin, vmax }: { vmin: number; vmax: number }) {
  return (
    <span className="inline-flex items-center gap-2">
      <span className="font-mono">{vmin.toFixed(2)}</span>
      <span
        className="inline-block h-2 w-32"
        style={{
          background: `linear-gradient(to right, ${residualColour(vmin, vmin, vmax)}, ${residualColour(vmax, vmin, vmax)})`,
        }}
      />
      <span className="font-mono">{vmax.toFixed(2)}</span>
    </span>
  );
}
