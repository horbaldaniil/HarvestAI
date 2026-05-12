import { useMemo } from "react";
import { AttributionControl, GeoJSON, MapContainer, TileLayer } from "react-leaflet";
import type { Feature, FeatureCollection } from "geojson";

import { useCoverage } from "@/hooks/useMethodology";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2 } from "lucide-react";

/**
 * Leaflet map showing where Sentinel-2 oblast sampling lives.
 *
 * 96 ~1km² sample polygons drawn on a CartoDB Positron basemap. Each oblast
 * gets a per-feature colour from a small categorical palette so the eye can
 * group samples by their source oblast.
 */
export function OblastCoverageMap() {
  const { data, isLoading } = useCoverage();

  const palette = useMemo(
    () => [
      "#5e7d36", "#7d9c49", "#bccf8e", "#3b82f6", "#a16207", "#dc2626",
      "#0ea5e9", "#16a34a", "#9333ea", "#f59e0b", "#0d9488", "#be185d",
    ],
    [],
  );

  const style = useMemo(() => {
    const oblastColour = new Map<string, string>();
    return (feature?: Feature) => {
      const ob = (feature?.properties as { oblast?: string } | undefined)?.oblast ?? "";
      if (!oblastColour.has(ob)) {
        oblastColour.set(ob, palette[oblastColour.size % palette.length]);
      }
      return {
        color: oblastColour.get(ob),
        weight: 2,
        fillColor: oblastColour.get(ob),
        fillOpacity: 0.5,
      };
    };
  }, [palette]);

  if (isLoading) {
    return (
      <Card>
        <CardContent className="flex h-80 items-center justify-center">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }

  const gjson = data?.samples_geojson as FeatureCollection | undefined;
  const hasSamples = (gjson?.features?.length ?? 0) > 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">
          Покриття областей зразками
          {hasSamples && (
            <span className="ml-2 text-xs font-normal text-muted-foreground">
              {gjson!.features.length} ~1км² зразків · {data?.oblasts.length ?? 0} областей
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <div className="h-[400px] w-full">
          <MapContainer
            center={[49.0, 31.5]}
            zoom={6}
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
            {hasSamples && (
              <GeoJSON
                key={JSON.stringify(gjson?.features.length)}
                data={gjson as FeatureCollection}
                style={style as never}
                onEachFeature={(feature, layer) => {
                  const props = feature.properties as { oblast?: string; sample_idx?: number };
                  const ob = props.oblast ?? "?";
                  const idx = props.sample_idx ?? "?";
                  layer.bindTooltip(`${ob} — sample ${idx}`, { sticky: true });
                }}
              />
            )}
          </MapContainer>
        </div>
        {!hasSamples && (
          <div className="border-t p-3 text-center text-sm text-muted-foreground">
            Зразки ще не згенеровано. Запустіть{" "}
            <code className="rounded bg-muted px-1">scripts/generate_oblast_samples.py</code>{" "}
            у backend.
          </div>
        )}
      </CardContent>
    </Card>
  );
}
