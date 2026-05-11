import { ImageOverlay } from "react-leaflet";
import type { LatLngBoundsLiteral } from "leaflet";
import type { FieldRead } from "@/api/fields";
import { type IndexName } from "@/api/observations";
import { useHeatmapBlob } from "@/hooks/useHeatmapBlob";

interface HeatmapOverlayProps {
  field: FieldRead;
  date: string;
  index: IndexName;
  opacity?: number;
}

/** Compute the polygon's lat-lng bbox so the PNG aligns with the field. */
function fieldBounds(field: FieldRead): LatLngBoundsLiteral {
  const ring = field.geometry.coordinates[0];
  let minLat = 90,
    maxLat = -90,
    minLng = 180,
    maxLng = -180;
  for (const [lng, lat] of ring) {
    if (lat < minLat) minLat = lat;
    if (lat > maxLat) maxLat = lat;
    if (lng < minLng) minLng = lng;
    if (lng > maxLng) maxLng = lng;
  }
  return [
    [minLat, minLng],
    [maxLat, maxLng],
  ];
}

export function HeatmapOverlay({
  field,
  date,
  index,
  opacity = 0.7,
}: HeatmapOverlayProps) {
  // Fetches PNG via axios (so JWT is attached) and exposes an object URL.
  const { data: blobUrl } = useHeatmapBlob(field.id, date, index);
  if (!blobUrl) return null;

  return (
    <ImageOverlay
      key={`${field.id}-${date}-${index}-${blobUrl}`}
      url={blobUrl}
      bounds={fieldBounds(field)}
      opacity={opacity}
      zIndex={450}
      // className flows onto the underlying <img>; the CSS rule in
      // globals.css disables interpolation so each Sentinel-2 pixel
      // renders as a sharp square instead of a blurred blob.
      className="heatmap-overlay-pixelated"
    />
  );
}
