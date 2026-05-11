import { type ReactNode } from "react";
import { LayersControl, MapContainer, TileLayer } from "react-leaflet";
import type { LatLngExpression } from "leaflet";
import { useTranslation } from "react-i18next";

// Center of Ukraine — used as initial map center.
export const UKRAINE_CENTER: LatLngExpression = [49.0, 31.5];
export const UKRAINE_DEFAULT_ZOOM = 6;

interface FieldMapProps {
  children?: ReactNode;
  center?: LatLngExpression;
  zoom?: number;
  className?: string;
}

export function FieldMap({
  children,
  center = UKRAINE_CENTER,
  zoom = UKRAINE_DEFAULT_ZOOM,
  className,
}: FieldMapProps) {
  const { t } = useTranslation();

  return (
    <MapContainer
      center={center}
      zoom={zoom}
      className={className ?? "h-full w-full"}
      scrollWheelZoom
      worldCopyJump
    >
      <LayersControl position="topright">
        <LayersControl.BaseLayer checked name={t("map.baseLayer.osm")}>
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            maxZoom={19}
          />
        </LayersControl.BaseLayer>
        <LayersControl.BaseLayer name={t("map.baseLayer.satellite")}>
          <TileLayer
            attribution="Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            maxZoom={19}
          />
        </LayersControl.BaseLayer>
      </LayersControl>
      {children}
    </MapContainer>
  );
}
