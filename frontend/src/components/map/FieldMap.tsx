import { type ReactNode } from "react";
import {
  AttributionControl,
  LayerGroup,
  LayersControl,
  MapContainer,
  TileLayer,
} from "react-leaflet";
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
      // Disable the default attribution control so we can replace it with
      // one that drops the "Leaflet |" prefix while keeping the tile-
      // provider credits (required by OSM ODbL + CartoDB ToS).
      attributionControl={false}
    >
      <AttributionControl prefix={false} position="bottomright" />
      <LayersControl position="topright">
        <LayersControl.BaseLayer checked name={t("map.baseLayer.osm")}>
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            maxZoom={19}
          />
        </LayersControl.BaseLayer>
        <LayersControl.BaseLayer name={t("map.baseLayer.satellite")}>
          {/*
            Hybrid satellite — Esri imagery + transparent reference layer
            on top. The reference layer ships country / oblast / raion
            boundaries plus city + village labels with a transparent
            background, so place names and borders read clearly over
            the imagery. Both tile layers are wrapped in a single
            LayerGroup so the LayersControl base-layer toggle treats
            them as one entity (turning satellite on shows both;
            turning it off hides both).
          */}
          <LayerGroup>
            <TileLayer
              attribution="Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"
              url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
              maxZoom={19}
            />
            <TileLayer
              // Esri reference overlay — administrative boundaries +
              // place names. Free, no API key. The PNG tiles are
              // transparent outside of labels/lines, so they composite
              // onto the imagery beneath without obscuring fields.
              attribution=""
              url="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
              maxZoom={19}
            />
          </LayerGroup>
        </LayersControl.BaseLayer>
      </LayersControl>
      {children}
    </MapContainer>
  );
}
