import { api } from "./client";

export interface User {
  id: number;
  email: string;
  full_name: string | null;
  locale: string;
  is_active: boolean;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface RegisterPayload {
  email: string;
  password: string;
  full_name?: string;
}

export interface LoginPayload {
  email: string;
  password: string;
}

export async function register(payload: RegisterPayload): Promise<User> {
  const { data } = await api.post<User>("/api/auth/register", payload);
  return data;
}

export async function login(payload: LoginPayload): Promise<TokenResponse> {
  const { data } = await api.post<TokenResponse>("/api/auth/login", payload);
  return data;
}

/**
 * Trade the httpOnly `harvestai_refresh` cookie for a fresh access
 * token + rotated refresh cookie. Called once on app start-up so a
 * page refresh doesn't force a re-login.
 *
 * Returns just the token shape — call `me()` afterwards to get the
 * user. The backend doesn't bundle them because the refresh endpoint
 * is sometimes called from the axios response-interceptor on a 401,
 * where we already have the user in memory.
 */
export async function refresh(): Promise<TokenResponse> {
  const { data } = await api.post<TokenResponse>("/api/auth/refresh");
  return data;
}

export async function logout(): Promise<void> {
  await api.post("/api/auth/logout");
}

export async function me(): Promise<User> {
  const { data } = await api.get<User>("/api/auth/me");
  return data;
}
