import { useEffect, useRef } from "react";
import { useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet-draw";
import type { Polygon as GJPolygon } from "geojson";
import { applyUkrainianDrawLocale } from "./leaflet-draw-locale";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type LDrawer = any;

interface PolygonDrawerProps {
  active: boolean;
  onComplete: (polygon: GJPolygon) => void;
  onCancel?: () => void;
  /** Receives the live drawer instance so the sidebar can offer a "Finish" button. */
  onDrawerReady?: (drawer: LDrawer | null) => void;
}

// ─── Defensive shim for leaflet-draw area readout ──────────────
//
// In Leaflet 1.9+ some Leaflet-draw versions assume L.GeometryUtil.readableArea
// exists. When it doesn't, an exception is thrown inside the click handler
// (after the third vertex, when area becomes computable) and silently breaks
// the click pipeline. We patch it once at module load.
function installGeometryUtilShim(): void {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const LAny = L as any;
  if (!LAny.GeometryUtil) LAny.GeometryUtil = {};
  if (typeof LAny.GeometryUtil.readableArea !== "function") {
    LAny.GeometryUtil.readableArea = (m2: number, metric: boolean): string => {
      if (metric) {
        return m2 >= 10000 ? `${(m2 / 10000).toFixed(2)} ha` : `${m2.toFixed(0)} m²`;
      }
      return `${m2.toFixed(2)}`;
    };
  }
  if (typeof LAny.GeometryUtil.geodesicArea !== "function") {
    LAny.GeometryUtil.geodesicArea = (latLngs: L.LatLng[]): number => {
      const R = 6_378_137; // WGS84 equatorial radius (m)
      const rad = Math.PI / 180;
      let area = 0;
      const n = latLngs.length;
      if (n < 3) return 0;
      for (let i = 0; i < n; i++) {
        const p1 = latLngs[i];
        const p2 = latLngs[(i + 1) % n];
        area +=
          (p2.lng - p1.lng) * rad *
          (2 + Math.sin(p1.lat * rad) + Math.sin(p2.lat * rad));
      }
      return Math.abs((area * R * R) / 2);
    };
  }
  if (typeof LAny.GeometryUtil.formattedNumber !== "function") {
    LAny.GeometryUtil.formattedNumber = (n: string | number, precision: number): string =>
      Number(n).toFixed(precision);
  }
}
installGeometryUtilShim();

/**
 * Draws a new polygon and reports it as GeoJSON upon completion.
 *
 * Callback props are intentionally NOT in the effect's dependency array. They
 * are inline functions on the parent (new identity every render), so depending
 * on them would tear down and re-create the leaflet-draw handler on every
 * parent re-render — losing the user's in-progress vertices.
 *
 * Completion is supported via three paths:
 *   1. Click on the first marker (leaflet-draw built-in)
 *   2. Double-click anywhere on the map (we wire this up explicitly because
 *      leaflet-draw does not do it; we also suspend `doubleClickZoom` so the
 *      map doesn't zoom on the second click)
 *   3. Imperative call via the drawer instance exposed through onDrawerReady
 *      (the sidebar uses this to render a "Finish" button)
 */
export function PolygonDrawer({
  active,
  onComplete,
  onCancel,
  onDrawerReady,
}: PolygonDrawerProps) {
  const map = useMap();
  const onCompleteRef = useRef(onComplete);
  const onCancelRef = useRef(onCancel);
  const onDrawerReadyRef = useRef(onDrawerReady);
  const completedRef = useRef(false);

  useEffect(() => {
    onCompleteRef.current = onComplete;
    onCancelRef.current = onCancel;
    onDrawerReadyRef.current = onDrawerReady;
  });

  useEffect(() => {
    applyUkrainianDrawLocale();
  }, []);

  useEffect(() => {
    if (!active) return;

    completedRef.current = false;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const LAny = L as any;
    if (!LAny.Draw?.Polygon) return;

    const drawer = new LAny.Draw.Polygon(map, {
      allowIntersection: false,
      showArea: false,
      showLength: false,
      shapeOptions: { color: "#7d9c49", weight: 2 },
    });
    drawer.enable();

    // Suspend zoom-on-double-click — we use double-click to finish drawing.
    const dczWasEnabled = map.doubleClickZoom.enabled();
    if (dczWasEnabled) map.doubleClickZoom.disable();

    onDrawerReadyRef.current?.(drawer);

    const onCreated = (e: L.LeafletEvent) => {
      completedRef.current = true;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const layer = (e as any).layer as L.Polygon;
      const feature = layer.toGeoJSON() as GeoJSON.Feature<GJPolygon>;
      onCompleteRef.current(feature.geometry);
    };

    const onDblClick = (e: L.LeafletMouseEvent) => {
      // Stop the underlying browser dblclick so nothing else fires.
      L.DomEvent.preventDefault(e.originalEvent);
      // Need ≥3 vertices for a polygon — leaflet-draw's _markers tracks them.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const markers = (drawer as any)._markers as L.Marker[] | undefined;
      if (markers && markers.length >= 3) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (drawer as any)._finishShape();
      }
    };

    map.on(LAny.Draw.Event.CREATED, onCreated);
    map.on("dblclick", onDblClick);

    return () => {
      map.off(LAny.Draw.Event.CREATED, onCreated);
      map.off("dblclick", onDblClick);
      drawer.disable();
      if (dczWasEnabled) map.doubleClickZoom.enable();
      onDrawerReadyRef.current?.(null);
      if (!completedRef.current) onCancelRef.current?.();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, map]);

  return null;
}
