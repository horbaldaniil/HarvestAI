import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { Loader2 } from "lucide-react";
import type { Polygon as GJPolygon } from "geojson";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { CropType, FieldRead } from "@/api/fields";

const CROPS: CropType[] = ["wheat", "corn", "sunflower"];

export interface FieldFormValues {
  name: string;
  crop_type: CropType;
  season_year: number;
}

interface FieldFormDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  mode: "create" | "edit";
  /** For create: the freshly drawn polygon (used to compute area preview). */
  pendingPolygon?: GJPolygon | null;
  /** For edit: the existing field. */
  field?: FieldRead | null;
  isSubmitting: boolean;
  onSubmit: (values: FieldFormValues) => void | Promise<void>;
}

export function FieldFormDialog({
  open,
  onOpenChange,
  mode,
  pendingPolygon,
  field,
  isSubmitting,
  onSubmit,
}: FieldFormDialogProps) {
  const { t } = useTranslation();
  const currentYear = new Date().getFullYear();

  const [name, setName] = useState("");
  const [cropType, setCropType] = useState<CropType>("wheat");
  const [seasonYear, setSeasonYear] = useState<number>(currentYear);

  // Reset form whenever the dialog opens with a different target.
  useEffect(() => {
    if (!open) return;
    if (mode === "edit" && field) {
      setName(field.name);
      setCropType(field.crop_type);
      setSeasonYear(field.season_year);
    } else {
      setName("");
      setCropType("wheat");
      setSeasonYear(currentYear);
    }
  }, [open, mode, field, currentYear]);

  // Lightweight client-side area estimate for the preview (shoelace + spherical
  // approximation). Real area comes from PostGIS after submit.
  const estimatedAreaHa = pendingPolygon ? estimateAreaHaSpherical(pendingPolygon) : null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await onSubmit({ name: name.trim(), crop_type: cropType, season_year: seasonYear });
  };

  const titleKey = mode === "create" ? "fields.create.title" : "fields.edit.title";
  const descriptionKey =
    mode === "create" ? "fields.create.description" : "fields.edit.description";
  const submitKey = mode === "create" ? "fields.create.submit" : "fields.edit.submit";

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{t(titleKey)}</DialogTitle>
          <DialogDescription>{t(descriptionKey)}</DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="field-name">{t("fields.name")}</Label>
            <Input
              id="field-name"
              required
              maxLength={255}
              placeholder={t("fields.namePlaceholder")}
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="field-crop">{t("fields.crop")}</Label>
            <Select
              value={cropType}
              onValueChange={(v) => setCropType(v as CropType)}
            >
              <SelectTrigger id="field-crop">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {CROPS.map((c) => (
                  <SelectItem key={c} value={c}>
                    {t(`fields.crops.${c}`)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="field-year">{t("fields.year")}</Label>
            <Input
              id="field-year"
              type="number"
              required
              min={2000}
              max={2100}
              value={seasonYear}
              onChange={(e) => setSeasonYear(Number(e.target.value))}
            />
          </div>

          {mode === "create" && estimatedAreaHa !== null && (
            <div className="rounded-md bg-muted px-3 py-2 text-sm">
              <span className="text-muted-foreground">{t("fields.area")}: </span>
              <span className="font-medium">
                ~{estimatedAreaHa.toFixed(2)} {t("fields.areaUnit")}
              </span>
            </div>
          )}

          {mode === "edit" && (
            <p className="text-xs text-muted-foreground">
              {t("fields.edit.geometryHint")}
            </p>
          )}

          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
              disabled={isSubmitting}
            >
              {t("common.cancel")}
            </Button>
            <Button type="submit" disabled={isSubmitting || !name.trim()}>
              {isSubmitting && <Loader2 className="h-4 w-4 animate-spin" />}
              {t(submitKey)}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

// ───── Local area estimate (for preview only) ──────────────────

const EARTH_RADIUS_M = 6_371_008.8;

function estimateAreaHaSpherical(polygon: GJPolygon): number {
  const ring = polygon.coordinates[0];
  if (!ring || ring.length < 4) return 0;
  let total = 0;
  for (let i = 0; i < ring.length - 1; i++) {
    const [lon1, lat1] = ring[i];
    const [lon2, lat2] = ring[i + 1];
    const dLon = (lon2 - lon1) * (Math.PI / 180);
    total +=
      dLon *
      (Math.sin((lat1 * Math.PI) / 180) + Math.sin((lat2 * Math.PI) / 180));
  }
  const m2 = (Math.abs(total) * EARTH_RADIUS_M * EARTH_RADIUS_M) / 2;
  return m2 / 10_000;
}
