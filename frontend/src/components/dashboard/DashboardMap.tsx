import { useMemo } from "react";
import {
  AttributionControl,
  GeoJSON,
  MapContainer,
  TileLayer,
  useMap,
} from "react-leaflet";
import type { FeatureCollection, Feature, Polygon } from "geojson";
import type { Layer, PathOptions } from "leaflet";
import { useEffect } from "react";
import { useNavigate } from "react-router-dom";

import { Card } from "@/components/ui/card";
import type { DashboardFieldRow } from "@/api/dashboard";

/**
 * Dashboard mini-map: thumbnail Leaflet showing every field polygon coloured
 * by latest NDVI. Click a field → /fields?selected=N (re-uses the deep-link
 * we already wired up). Empty state (no fields) shows a hint.
 */
export function DashboardMap({ fields }: { fields: DashboardFieldRow[] }) {
  const navigate = useNavigate();

  const fc = useMemo<FeatureCollection | null>(() => {
    const features: Feature[] = [];
    for (const f of fields) {
      if (!f.geometry) continue;
      features.push({
        type: "Feature",
        properties: {
          field_id: f.field_id,
          name: f.name,
          crop_type: f.crop_type,
          ndvi: f.current_ndvi,
          risk_score: f.risk_score,
        },
        geometry: f.geometry as Polygon,
      });
    }
    return features.length > 0 ? { type: "FeatureCollection", features } : null;
  }, [fields]);

  if (!fc) {
    return (
      <Card className="flex h-72 items-center justify-center text-sm text-muted-foreground">
        Додайте перше поле на сторінці «Поля», щоб побачити мапу.
      </Card>
    );
  }

  return (
    <Card className="overflow-hidden p-0">
      <div className="h-72 w-full">
        <MapContainer
          // Ukraine bounding centroid; AutoFit below pans to actual data.
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
          <GeoJSON
            key={fc.features.length}
            data={fc}
            style={ndviStyle as never}
            onEachFeature={(feat: Feature, layer: Layer) => {
              const p = feat.properties as {
                field_id: number;
                name: string;
                ndvi: number | null;
                risk_score: number;
              };
              const ndviStr =
                p.ndvi !== null && p.ndvi !== undefined
                  ? `NDVI ${p.ndvi.toFixed(2)}`
                  : "NDVI —";
              layer.bindTooltip(
                `<strong>${p.name}</strong><br/>${ndviStr} · risk ${p.risk_score}`,
                { sticky: true, direction: "top" },
              );
              layer.on("click", () => navigate(`/fields?selected=${p.field_id}`));
            }}
          />
          <FitToFeatures fc={fc} />
        </MapContainer>
      </div>
    </Card>
  );
}

function FitToFeatures({ fc }: { fc: FeatureCollection }) {
  const map = useMap();
  useEffect(() => {
    if (!fc.features.length) return;
    import("leaflet").then((L) => {
      // GeoJSON layer just to compute bounds — never added to map.
      const layer = L.geoJSON(fc as never);
      const bounds = layer.getBounds();
      if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [20, 20], maxZoom: 13 });
      }
    });
  }, [fc, map]);
  return null;
}

/**
 * NDVI gradient: red (≤0.3) → yellow (≈0.5) → green (≥0.7).
 * Returns full Leaflet PathOptions so we can also tweak weight.
 */
function ndviStyle(feature?: Feature): PathOptions {
  const ndvi = (feature?.properties as { ndvi?: number | null } | undefined)?.ndvi;
  if (ndvi === null || ndvi === undefined) {
    return { color: "#9ca3af", weight: 2, fillColor: "#d1d5db", fillOpacity: 0.5 };
  }
  // Smooth piecewise linear interpolation between three stops.
  // Below 0.3 → red, around 0.5 → amber, above 0.7 → green.
  const colour = ndvi >= 0.7
    ? "#16a34a"
    : ndvi >= 0.5
    ? "#84cc16"
    : ndvi >= 0.3
    ? "#facc15"
    : "#dc2626";
  return { color: colour, weight: 2, fillColor: colour, fillOpacity: 0.55 };
}
