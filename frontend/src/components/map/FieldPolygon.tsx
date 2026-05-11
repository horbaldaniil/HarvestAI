import { Polygon, Tooltip } from "react-leaflet";
import type { LatLngExpression } from "leaflet";
import type { FieldRead } from "@/api/fields";
import { cropColor } from "@/lib/colors";

interface FieldPolygonProps {
  field: FieldRead;
  selected?: boolean;
  onClick?: (field: FieldRead) => void;
}

// GeoJSON: [lon, lat]. Leaflet: [lat, lng].
function geoJsonRingToLeaflet(ring: number[][]): LatLngExpression[] {
  return ring.map(([lon, lat]) => [lat, lon]);
}

export function FieldPolygon({ field, selected = false, onClick }: FieldPolygonProps) {
  const color = cropColor(field.crop_type, field.color);
  const positions = field.geometry.coordinates.map(geoJsonRingToLeaflet);

  return (
    <Polygon
      positions={positions}
      pathOptions={{
        color: selected ? "#1f2937" : color,
        weight: selected ? 3 : 2,
        fillColor: color,
        fillOpacity: selected ? 0.55 : 0.35,
      }}
      eventHandlers={{
        click: () => onClick?.(field),
      }}
    >
      <Tooltip direction="top" sticky>
        <div className="text-sm">
          <div className="font-semibold">{field.name}</div>
          <div className="text-xs text-gray-600">
            {field.area_ha.toFixed(2)} га · {field.season_year}
          </div>
        </div>
      </Tooltip>
    </Polygon>
  );
}
