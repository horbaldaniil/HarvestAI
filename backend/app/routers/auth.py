"""Auth endpoints: register / login / refresh / logout / me.

Token delivery:
- Access token — in JSON response body; frontend stores in memory + Authorization header.
- Refresh token — in httpOnly secure SameSite=Lax cookie; the only place it can survive XSS.
"""
from __future__ import annotations

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status

from app.config import settings
from app.deps import CurrentUser, DbSession
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserPublic
from app.services import auth_service
from app.utils.security import create_access_token

router = APIRouter(prefix="/api/auth", tags=["auth"])

REFRESH_COOKIE = "harvestai_refresh"


def _set_refresh_cookie(response: Response, refresh_token: str) -> None:
    response.set_cookie(
        key=REFRESH_COOKIE,
        value=refresh_token,
        httponly=True,
        secure=settings.is_production,  # require HTTPS only in prod
        samesite="lax",
        max_age=settings.jwt_refresh_expire_days * 24 * 60 * 60,
        path="/api/auth",
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(key=REFRESH_COOKIE, path="/api/auth")


@router.post("/register", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
async def register(data: RegisterRequest, db: DbSession) -> UserPublic:
    user = await auth_service.register_user(db, data)
    return UserPublic.model_validate(user)


@router.post("/login", response_model=TokenResponse)
async def login(data: LoginRequest, response: Response, db: DbSession) -> TokenResponse:
    user = await auth_service.authenticate_user(db, str(data.email), data.password)
    access = create_access_token(subject=user.id)
    refresh = await auth_service.issue_refresh_token(db, user.id)
    _set_refresh_cookie(response, refresh)
    return TokenResponse(
        access_token=access,
        expires_in=settings.jwt_access_expire_minutes * 60,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    response: Response,
    db: DbSession,
    harvestai_refresh: str | None = Cookie(default=None, alias=REFRESH_COOKIE),
) -> TokenResponse:
    if not harvestai_refresh:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token відсутній.",
        )
    user, new_refresh = await auth_service.rotate_refresh_token(db, harvestai_refresh)
    access = create_access_token(subject=user.id)
    _set_refresh_cookie(response, new_refresh)
    return TokenResponse(
        access_token=access,
        expires_in=settings.jwt_access_expire_minutes * 60,
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    response: Response,
    db: DbSession,
    harvestai_refresh: str | None = Cookie(default=None, alias=REFRESH_COOKIE),
) -> Response:
    if harvestai_refresh:
        await auth_service.revoke_refresh_token(db, harvestai_refresh)
    _clear_refresh_cookie(response)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/me", response_model=UserPublic)
async def me(current_user: CurrentUser) -> UserPublic:
    return UserPublic.model_validate(current_user)
