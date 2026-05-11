import { useEffect, useMemo, useRef } from "react";
import { FeatureGroup, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet-draw";
import type { Polygon as GJPolygon } from "geojson";
import { applyUkrainianDrawLocale } from "./leaflet-draw-locale";
import { cropColor } from "@/lib/colors";
import type { FieldRead } from "@/api/fields";

interface PolygonEditorProps {
  field: FieldRead;
  onSave: (polygon: GJPolygon) => void;
  onCancel: () => void;
}

/**
 * Wraps a single field polygon with draggable edit handles. When the user
 * confirms (via the "Save" button in the toolbar) we read out the modified
 * GeoJSON and bubble it up.
 */
export function PolygonEditor({ field, onSave, onCancel }: PolygonEditorProps) {
  const map = useMap();
  const groupRef = useRef<L.FeatureGroup>(null);
  const layerRef = useRef<L.Polygon | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const editorRef = useRef<any>(null);

  const positions = useMemo(
    () =>
      field.geometry.coordinates.map((ring) => ring.map(([lon, lat]) => [lat, lon])),
    [field],
  );

  useEffect(() => {
    applyUkrainianDrawLocale();
  }, []);

  useEffect(() => {
    if (!groupRef.current) return;

    const color = cropColor(field.crop_type, field.color);
    const layer = L.polygon(positions as L.LatLngExpression[][], {
      color: "#1f2937",
      weight: 3,
      fillColor: color,
      fillOpacity: 0.4,
    });
    groupRef.current.addLayer(layer);
    layerRef.current = layer;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const editHandler = new (L as any).EditToolbar.Edit(map, {
      featureGroup: groupRef.current,
      selectedPathOptions: { dashArray: "10, 10", fillOpacity: 0.4 },
    });
    editHandler.enable();
    editorRef.current = editHandler;

    const onEdited = () => {
      const geo = layer.toGeoJSON() as GeoJSON.Feature<GJPolygon>;
      onSave(geo.geometry);
    };
    const onEditCancel = () => {
      onCancel();
    };

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    map.on((L as any).Draw.Event.EDITSTOP, onEdited);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    map.on((L as any).Draw.Event.DELETESTOP, onEditCancel);

    return () => {
      editHandler.disable();
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      map.off((L as any).Draw.Event.EDITSTOP, onEdited);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      map.off((L as any).Draw.Event.DELETESTOP, onEditCancel);
      groupRef.current?.removeLayer(layer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [field.id]);

  return <FeatureGroup ref={groupRef} />;
}
