"""Fields router: CRUD endpoints for user-owned agricultural parcels.

All endpoints require an authenticated user; ownership is enforced inside the
service layer (cross-user reads return 404, not 403, to avoid existence leaks).
"""
from __future__ import annotations

from fastapi import APIRouter, status

from app.deps import CurrentUser, DbSession
from app.schemas.field import FieldCreate, FieldRead, FieldUpdate
from app.services import field_service

router = APIRouter(prefix="/api/fields", tags=["fields"])


@router.get("", response_model=list[FieldRead])
async def list_fields(current_user: CurrentUser, db: DbSession) -> list[FieldRead]:
    return await field_service.list_fields(db, current_user.id)


@router.post("", response_model=FieldRead, status_code=status.HTTP_201_CREATED)
async def create_field(
    data: FieldCreate, current_user: CurrentUser, db: DbSession
) -> FieldRead:
    return await field_service.create_field(db, current_user.id, data)


@router.get("/{field_id}", response_model=FieldRead)
async def get_field(
    field_id: int, current_user: CurrentUser, db: DbSession
) -> FieldRead:
    return await field_service.get_field(db, current_user.id, field_id)


@router.patch("/{field_id}", response_model=FieldRead)
async def update_field(
    field_id: int, data: FieldUpdate, current_user: CurrentUser, db: DbSession
) -> FieldRead:
    return await field_service.update_field(db, current_user.id, field_id, data)


@router.delete("/{field_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_field(
    field_id: int, current_user: CurrentUser, db: DbSession
) -> None:
    await field_service.delete_field(db, current_user.id, field_id)
