import { format } from "date-fns";
import { uk } from "date-fns/locale";
import { Download, FileText, Loader2, Trash2 } from "lucide-react";

import {
  triggerBrowserDownload,
  type GeneratedReportRead,
} from "@/api/reports";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  useDeleteReport,
  useDownloadSavedReport,
  useReports,
} from "@/hooks/useReports";

const KIND_LABEL: Record<GeneratedReportRead["kind"], string> = {
  field: "Поле",
  portfolio: "Портфель",
  compare: "Порівняння",
};

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function ReportsHistory() {
  const { data, isLoading } = useReports();
  const download = useDownloadSavedReport();
  const remove = useDeleteReport();

  const handleDownload = async (r: GeneratedReportRead) => {
    const blob = await download.mutateAsync(r.id);
    triggerBrowserDownload(blob, `${r.title.replace(/[^\p{L}\p{N}_-]+/gu, "_")}.pdf`);
  };

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }
  if (!data || data.length === 0) {
    return (
      <Card className="py-12 text-center text-sm text-muted-foreground">
        <FileText className="mx-auto mb-2 h-6 w-6 opacity-50" />
        Звітів ще немає. Згенеруйте перший на вкладці Builder.
      </Card>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b text-left text-xs text-muted-foreground">
            <th className="px-3 py-2 font-medium">Назва</th>
            <th className="px-3 py-2 font-medium">Тип</th>
            <th className="px-3 py-2 font-medium">Дата</th>
            <th className="px-3 py-2 text-right font-medium">Розмір</th>
            <th className="px-3 py-2"></th>
          </tr>
        </thead>
        <tbody>
          {data.map((r) => (
            <tr key={r.id} className="border-b last:border-b-0 hover:bg-accent/30">
              <td className="px-3 py-2 font-medium">{r.title}</td>
              <td className="px-3 py-2 text-muted-foreground">{KIND_LABEL[r.kind]}</td>
              <td className="px-3 py-2 text-muted-foreground">
                {format(new Date(r.generated_at), "d MMM yyyy, HH:mm", { locale: uk })}
              </td>
              <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                {formatSize(r.file_size_bytes)}
              </td>
              <td className="px-3 py-2 text-right">
                <Button
                  size="icon"
                  variant="ghost"
                  onClick={() => handleDownload(r)}
                  disabled={download.isPending}
                  aria-label="Завантажити"
                >
                  <Download className="h-4 w-4" />
                </Button>
                <Button
                  size="icon"
                  variant="ghost"
                  onClick={() => remove.mutate(r.id)}
                  disabled={remove.isPending}
                  aria-label="Видалити"
                >
                  <Trash2 className="h-4 w-4 text-muted-foreground" />
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-3 text-xs text-muted-foreground">
        Зберігається до 50 останніх звітів на користувача. Старіші
        видаляються автоматично.
      </p>
    </div>
  );
}
