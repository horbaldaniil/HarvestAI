import { api } from "./client";

export type ReportKind = "field" | "portfolio" | "compare";
export type FieldSection =
  | "summary"
  | "indices"
  | "prediction"
  | "alerts"
  | "forecast";

export const ALL_FIELD_SECTIONS: FieldSection[] = [
  "summary",
  "indices",
  "prediction",
  "alerts",
  "forecast",
];

export const SECTION_LABELS: Record<FieldSection, string> = {
  summary: "AI-резюме + heatmap",
  indices: "Графіки NDVI / EVI / NDWI / SAVI",
  prediction: "Прогноз врожайності + SHAP",
  alerts: "Аномалії",
  forecast: "Прогноз погоди (14 днів)",
};

export interface GeneratedReportRead {
  id: number;
  kind: ReportKind;
  title: string;
  params_json: Record<string, unknown>;
  file_size_bytes: number;
  generated_at: string;
}

export interface BuilderPayload {
  kind: ReportKind;
  field_id?: number;
  field_ids?: number[];
  sections?: FieldSection[];
  date_from?: string;
  date_to?: string;
  title?: string;
}

/**
 * Trigger a per-field PDF download. Returns the Blob so the caller can hand
 * it to `triggerBrowserDownload` or transform further (e.g. preview iframe).
 */
export async function downloadFieldReport(fieldId: number): Promise<Blob> {
  const { data } = await api.post<Blob>(
    `/api/fields/${fieldId}/report`,
    undefined,
    { responseType: "blob" },
  );
  return data;
}

export async function downloadPortfolioReport(): Promise<Blob> {
  const { data } = await api.post<Blob>(
    `/api/portfolio/report`,
    undefined,
    { responseType: "blob" },
  );
  return data;
}

/**
 * Run the builder endpoint with custom params and stream back the PDF.
 */
export async function runReportBuilder(payload: BuilderPayload): Promise<Blob> {
  const { data } = await api.post<Blob>("/api/reports/builder", payload, {
    responseType: "blob",
  });
  return data;
}

export async function listReports(): Promise<GeneratedReportRead[]> {
  const { data } = await api.get<GeneratedReportRead[]>("/api/reports");
  return data;
}

export async function downloadSavedReport(reportId: number): Promise<Blob> {
  const { data } = await api.get<Blob>(
    `/api/reports/${reportId}/download`,
    { responseType: "blob" },
  );
  return data;
}

export async function deleteReport(reportId: number): Promise<void> {
  await api.delete(`/api/reports/${reportId}`);
}

/**
 * Browser-side helper: create a temporary object URL, click an invisible
 * anchor, then revoke the URL. Works in every major browser.
 */
export function triggerBrowserDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Revoke shortly after — Firefox needs the URL to stay alive for a tick.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
