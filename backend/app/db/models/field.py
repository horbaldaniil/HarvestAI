"""Field ORM model (user-drawn agricultural parcel).

Geometry is stored as PostGIS GEOGRAPHY(POLYGON, 4326). We use GEOGRAPHY (not
GEOMETRY) because ST_Area on GEOGRAPHY returns square meters directly — no
projection juggling needed — which makes the generated `area_ha` column
trivially correct on WGS84 inputs.

`area_ha` and `centroid` are GENERATED columns so they cannot drift from the
polygon: clients never write them, the database derives them.
"""
from __future__ import annotations

from geoalchemy2 import Geography
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Computed,
    Enum,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin
from app.db.models.enums import CropType


class Field(Base, TimestampMixin):
    __tablename__ = "fields"
    __table_args__ = (
        CheckConstraint("season_year BETWEEN 2000 AND 2100", name="season_year_range"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # Use PostgreSQL ENUM (matches migration 0002). values_callable lets us
    # store the Python enum's value (lowercase string) instead of the member name.
    crop_type: Mapped[CropType] = mapped_column(
        Enum(
            CropType,
            name="crop_type_enum",
            values_callable=lambda enum: [e.value for e in enum],
            create_type=False,  # type created by migration 0002
        ),
        nullable=False,
    )
    season_year: Mapped[int] = mapped_column(Integer, nullable=False)
    color: Mapped[str | None] = mapped_column(String(7), nullable=True)

    # Polygon stored in WGS84 (EPSG:4326).
    geom: Mapped[object] = mapped_column(
        Geography(geometry_type="POLYGON", srid=4326, spatial_index=True),
        nullable=False,
    )

    # Generated columns — DB-side derivations of geom.
    # Use ::geometry cast for ST_Centroid (no GEOGRAPHY overload).
    area_ha: Mapped[float] = mapped_column(
        Computed("ST_Area(geom) / 10000.0", persisted=True),
    )
    centroid: Mapped[object] = mapped_column(
        Geography(geometry_type="POINT", srid=4326),
        Computed("ST_Centroid(geom::geometry)::geography", persisted=True),
    )

    def __repr__(self) -> str:
        return f"<Field id={self.id} name={self.name!r} crop={self.crop_type}>"
