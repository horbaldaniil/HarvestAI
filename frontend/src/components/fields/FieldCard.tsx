import { useTranslation } from "react-i18next";
import { Pencil, Trash2 } from "lucide-react";
import type { FieldRead } from "@/api/fields";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cropColor } from "@/lib/colors";
import { cn } from "@/lib/utils";

interface FieldCardProps {
  field: FieldRead;
  selected?: boolean;
  onSelect: (field: FieldRead) => void;
  onEdit: (field: FieldRead) => void;
  onDelete: (field: FieldRead) => void;
}

export function FieldCard({
  field,
  selected = false,
  onSelect,
  onEdit,
  onDelete,
}: FieldCardProps) {
  const { t } = useTranslation();
  const color = cropColor(field.crop_type, field.color);

  return (
    <div
      className={cn(
        "group cursor-pointer rounded-lg border p-3 transition-colors",
        selected
          ? "border-primary bg-primary/5"
          : "border-border bg-card hover:bg-accent/40",
      )}
      onClick={() => onSelect(field)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onSelect(field);
        }
      }}
    >
      <div className="flex items-start gap-2">
        <span
          className="mt-1 inline-block h-3 w-3 flex-shrink-0 rounded-sm"
          style={{ background: color }}
          aria-hidden
        />
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-semibold">{field.name}</div>
          <div className="mt-1 flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
            <Badge variant="secondary" className="font-normal">
              {t(`fields.crops.${field.crop_type}`)}
            </Badge>
            <span>·</span>
            <span>
              {field.area_ha.toFixed(2)} {t("fields.areaUnit")}
            </span>
            <span>·</span>
            <span>{field.season_year}</span>
          </div>
        </div>
      </div>

      {/* Action row — always visible (no hover-reveal). The MapPin
          "focus on map" button was removed because clicking the card
          itself already triggers `onSelect`, so the icon was a dead
          duplicate of the parent click handler. */}
      <div className="mt-2 flex gap-1">
        <Button
          variant="ghost"
          size="sm"
          className="h-7 flex-1 text-xs"
          onClick={(e) => {
            e.stopPropagation();
            onEdit(field);
          }}
          title={t("fields.actions.edit")}
        >
          <Pencil className="h-3.5 w-3.5" />
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className="h-7 flex-1 text-xs text-destructive hover:text-destructive"
          onClick={(e) => {
            e.stopPropagation();
            onDelete(field);
          }}
          title={t("fields.actions.delete")}
        >
          <Trash2 className="h-3.5 w-3.5" />
        </Button>
      </div>
    </div>
  );
}
