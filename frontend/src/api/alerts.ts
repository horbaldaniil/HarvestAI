import { api } from "./client";

export type AlertSeverity = "info" | "warning" | "critical";
export type AlertType = "ndvi_drop" | "drought_stress" | "heat_stress";

export interface AlertRead {
  id: number;
  field_id: number;
  field_name: string | null;
  severity: AlertSeverity;
  type: AlertType;
  message_uk: string;
  metric_value: number | null;
  threshold: number | null;
  acknowledged: boolean;
  created_at: string;
}

export async function listAlerts(acknowledged = false): Promise<AlertRead[]> {
  const { data } = await api.get<AlertRead[]>("/api/alerts", {
    params: { acknowledged, limit: 20 },
  });
  return data;
}

export async function alertsCount(): Promise<{ unread: number }> {
  const { data } = await api.get<{ unread: number }>("/api/alerts/count");
  return data;
}

export async function acknowledgeAlert(alertId: number): Promise<void> {
  await api.post(`/api/alerts/${alertId}/ack`);
}
