import { api } from "./client";

export interface QuotaResponse {
  units_used: number;
  units_limit: number;
  units_remaining: number;
  percent_used: number;
  period_start: string;
  period_end: string;
  refresh_at: string;
}

export async function getQuota(): Promise<QuotaResponse> {
  const { data } = await api.get<QuotaResponse>("/api/quota");
  return data;
}
