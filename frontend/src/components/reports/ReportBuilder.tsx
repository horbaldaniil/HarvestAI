import { useState } from "react";
import { Download, Loader2 } from "lucide-react";
import { toast } from "sonner";

import {
  ALL_FIELD_SECTIONS,
  SECTION_LABELS,
  triggerBrowserDownload,
  type BuilderPayload,
  type FieldSection,
  type ReportKind,
} from "@/api/reports";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useFields } from "@/hooks/useFields";
import { useRunBuilder } from "@/hooks/useReports";
import { cn } from "@/lib/utils";

/**
 * Builder form for customised PDF reports.
 *
 * Top-level tabs select the report shape; each shape exposes its own
 * subset of controls. We don't render hidden fields' state — switching
 * tabs resets per-tab inputs so the user can't accidentally submit a
 * mixed payload.
 */
export function ReportBuilder() {
  const [kind, setKind] = useState<ReportKind>("field");

  return (
    <Tabs
      defaultValue="field"
      value={kind}
      onValueChange={(v) => setKind(v as ReportKind)}
    >
      <TabsList>
        <TabsTrigger value="field">Одне поле</TabsTrigger>
        <TabsTrigger value="portfolio">Портфель</TabsTrigger>
        <TabsTrigger value="compare">Порівняти поля</TabsTrigger>
      </TabsList>

      <TabsContent value="field">
        <FieldBuilder />
      </TabsContent>
      <TabsContent value="portfolio">
        <PortfolioBuilder />
      </TabsContent>
      <TabsContent value="compare">
        <CompareBuilder />
      </TabsContent>
    </Tabs>
  );
}


// ─── Single-field builder ────────────────────────────────────


function FieldBuilder() {
  const fieldsQuery = useFields();
  const run = useRunBuilder();
  const [fieldId, setFieldId] = useState<string>("");
  const [sections, setSections] = useState<Set<FieldSection>>(
    new Set(ALL_FIELD_SECTIONS),
  );
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");

  const toggleSection = (s: FieldSection) => {
    setSections((curr) => {
      const next = new Set(curr);
      next.has(s) ? next.delete(s) : next.add(s);
      return next;
    });
  };

  const handleSubmit = async () => {
    if (!fieldId) {
      toast.error("Виберіть поле.");
      return;
    }
    const payload: BuilderPayload = {
      kind: "field",
      field_id: Number(fieldId),
      sections: Array.from(sections),
      date_from: dateFrom || undefined,
      date_to: dateTo || undefined,
    };
    const blob = await run.mutateAsync(payload);
    const fld = fieldsQuery.data?.find((f) => f.id === Number(fieldId));
    const safe = (fld?.name ?? "field").replace(/[^\p{L}\p{N}_-]+/gu, "_");
    triggerBrowserDownload(blob, `HarvestAI_${safe}_custom.pdf`);
    toast.success("Звіт згенеровано");
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Звіт по одному полю</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="field-pick">Поле</Label>
            <Select value={fieldId} onValueChange={setFieldId}>
              <SelectTrigger id="field-pick">
                <SelectValue placeholder="Виберіть поле" />
              </SelectTrigger>
              <SelectContent>
                {(fieldsQuery.data ?? []).map((f) => (
                  <SelectItem key={f.id} value={String(f.id)}>
                    {f.name} · {f.season_year}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="date-from">Від дати (опц.)</Label>
            <Input
              id="date-from"
              type="date"
              value={dateFrom}
              onChange={(e) => setDateFrom(e.target.value)}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="date-to">До дати (опц.)</Label>
            <Input
              id="date-to"
              type="date"
              value={dateTo}
              onChange={(e) => setDateTo(e.target.value)}
            />
          </div>
        </div>

        <div className="space-y-2">
          <Label>Секції у звіті</Label>
          <div className="grid gap-2 sm:grid-cols-2">
            {ALL_FIELD_SECTIONS.map((s) => (
              <SectionCheckbox
                key={s}
                section={s}
                checked={sections.has(s)}
                onChange={() => toggleSection(s)}
              />
            ))}
          </div>
        </div>

        <SubmitRow
          isPending={run.isPending}
          onSubmit={handleSubmit}
          disabled={!fieldId || sections.size === 0}
        />
      </CardContent>
    </Card>
  );
}


// ─── Portfolio builder (no params besides title) ─────────────


function PortfolioBuilder() {
  const run = useRunBuilder();

  const handleSubmit = async () => {
    const blob = await run.mutateAsync({ kind: "portfolio" });
    const stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, "");
    triggerBrowserDownload(blob, `HarvestAI_portfolio_${stamp}.pdf`);
    toast.success("Звіт згенеровано");
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Звіт по всьому портфелю</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="mb-4 text-sm text-muted-foreground">
          Зведена сторінка по всіх ваших полях: KPI, таблиця полів,
          порівняння рік-до-року, активні аномалії.
        </p>
        <SubmitRow isPending={run.isPending} onSubmit={handleSubmit} />
      </CardContent>
    </Card>
  );
}


// ─── Compare-fields builder ──────────────────────────────────


function CompareBuilder() {
  const fieldsQuery = useFields();
  const run = useRunBuilder();
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");

  const toggleField = (id: number) => {
    setSelected((curr) => {
      const next = new Set(curr);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const handleSubmit = async () => {
    if (selected.size < 2) {
      toast.error("Виберіть мінімум 2 поля.");
      return;
    }
    const blob = await run.mutateAsync({
      kind: "compare",
      field_ids: Array.from(selected),
      date_from: dateFrom || undefined,
      date_to: dateTo || undefined,
    });
    const stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, "");
    triggerBrowserDownload(blob, `HarvestAI_compare_${stamp}.pdf`);
    toast.success("Звіт згенеровано");
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Порівняння кількох полів</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <Label>Поля для порівняння (≥ 2)</Label>
          <div className="mt-2 grid gap-2 sm:grid-cols-2">
            {(fieldsQuery.data ?? []).map((f) => (
              <FieldCheckbox
                key={f.id}
                label={`${f.name} · ${f.season_year}`}
                checked={selected.has(f.id)}
                onChange={() => toggleField(f.id)}
              />
            ))}
          </div>
          {fieldsQuery.data?.length === 0 && (
            <p className="text-sm text-muted-foreground">У вас ще немає полів.</p>
          )}
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="cmp-date-from">Від дати (опц.)</Label>
            <Input
              id="cmp-date-from"
              type="date"
              value={dateFrom}
              onChange={(e) => setDateFrom(e.target.value)}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="cmp-date-to">До дати (опц.)</Label>
            <Input
              id="cmp-date-to"
              type="date"
              value={dateTo}
              onChange={(e) => setDateTo(e.target.value)}
            />
          </div>
        </div>

        <SubmitRow
          isPending={run.isPending}
          onSubmit={handleSubmit}
          disabled={selected.size < 2}
        />
      </CardContent>
    </Card>
  );
}


// ─── Small reusable bits ─────────────────────────────────────


function SectionCheckbox({
  section,
  checked,
  onChange,
}: {
  section: FieldSection;
  checked: boolean;
  onChange: () => void;
}) {
  return (
    <label
      className={cn(
        "flex cursor-pointer items-start gap-2 rounded-md border p-2 text-sm transition-colors",
        checked ? "border-primary bg-primary/5" : "hover:bg-accent/30",
      )}
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        className="mt-0.5"
      />
      <span>{SECTION_LABELS[section]}</span>
    </label>
  );
}


function FieldCheckbox({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: () => void;
}) {
  return (
    <label
      className={cn(
        "flex cursor-pointer items-center gap-2 rounded-md border p-2 text-sm transition-colors",
        checked ? "border-primary bg-primary/5" : "hover:bg-accent/30",
      )}
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
      />
      <span>{label}</span>
    </label>
  );
}


function SubmitRow({
  isPending,
  onSubmit,
  disabled,
}: {
  isPending: boolean;
  onSubmit: () => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex justify-end">
      <Button onClick={onSubmit} disabled={isPending || disabled}>
        {isPending ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <Download className="h-4 w-4" />
        )}
        {isPending ? "Генерую..." : "Згенерувати PDF"}
      </Button>
    </div>
  );
}
