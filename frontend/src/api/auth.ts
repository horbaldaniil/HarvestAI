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

export async function logout(): Promise<void> {
  await api.post("/api/auth/logout");
}

export async function me(): Promise<User> {
  const { data } = await api.get<User>("/api/auth/me");
  return data;
}
