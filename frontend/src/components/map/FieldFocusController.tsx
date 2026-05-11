import { useEffect } from "react";
import { useMap } from "react-leaflet";
import type { FieldRead } from "@/api/fields";

interface FieldFocusControllerProps {
  field: FieldRead | null | undefined;
  zoom?: number;
}

/**
 * Programmatic camera control: when a field becomes the "focused" one in the
 * sidebar, fly the map to its centroid. Lives as a child of <MapContainer>
 * so it can grab the map instance via useMap().
 */
export function FieldFocusController({ field, zoom = 14 }: FieldFocusControllerProps) {
  const map = useMap();

  useEffect(() => {
    if (!field) return;
    const [lon, lat] = field.centroid.coordinates;
    map.flyTo([lat, lon], zoom, { duration: 0.8 });
  }, [field, map, zoom]);

  return null;
}
