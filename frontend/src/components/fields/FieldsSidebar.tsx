import { useTranslation } from "react-i18next";
import { Check, Loader2, Plus, Sprout, X } from "lucide-react";
import type { FieldRead } from "@/api/fields";
import { Button } from "@/components/ui/button";
import { FieldCard } from "./FieldCard";

interface FieldsSidebarProps {
  fields: FieldRead[];
  isLoading: boolean;
  selectedId: number | null;
  drawing: boolean;
  /** True only when there are enough vertices on the in-progress polygon. */
  canFinishDrawing?: boolean;
  onStartDraw: () => void;
  onCancelDraw: () => void;
  onFinishDraw?: () => void;
  onSelect: (field: FieldRead) => void;
  onEdit: (field: FieldRead) => void;
  onDelete: (field: FieldRead) => void;
}

export function FieldsSidebar({
  fields,
  isLoading,
  selectedId,
  drawing,
  canFinishDrawing = false,
  onStartDraw,
  onCancelDraw,
  onFinishDraw,
  onSelect,
  onEdit,
  onDelete,
}: FieldsSidebarProps) {
  const { t } = useTranslation();

  return (
    <aside className="flex h-full w-80 flex-shrink-0 flex-col border-r bg-card">
      <header className="border-b p-4">
        <h2 className="text-lg font-semibold tracking-tight">{t("fields.title")}</h2>
        <p className="mt-0.5 text-xs text-muted-foreground">
          {fields.length} {t("fields.area").toLowerCase()}
        </p>
      </header>

      <div className="space-y-2 border-b p-3">
        {drawing ? (
          <>
            <Button
              className="w-full"
              onClick={onFinishDraw}
              disabled={!canFinishDrawing}
            >
              <Check className="h-4 w-4" />
              {t("fields.actions.finishDrawing")}
            </Button>
            <Button variant="outline" className="w-full" onClick={onCancelDraw}>
              <X className="h-4 w-4" />
              {t("fields.actions.cancelDrawing")}
            </Button>
          </>
        ) : (
          <Button className="w-full" onClick={onStartDraw}>
            <Plus className="h-4 w-4" />
            {t("fields.addNew")}
          </Button>
        )}
      </div>

      <div className="flex-1 overflow-y-auto">
        {isLoading ? (
          <div className="flex items-center justify-center p-8">
            <Loader2 className="h-5 w-5 animate-spin text-primary" />
          </div>
        ) : fields.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-3 p-8 text-center">
            <div className="flex h-14 w-14 items-center justify-center rounded-full bg-harvest-100 text-harvest-600">
              <Sprout className="h-7 w-7" />
            </div>
            <div>
              <h3 className="text-sm font-semibold">{t("fields.empty.title")}</h3>
              <p className="mt-1 text-xs text-muted-foreground">
                {t("fields.empty.description")}
              </p>
            </div>
            <Button size="sm" variant="default" onClick={onStartDraw} className="mt-1">
              <Plus className="h-3.5 w-3.5" />
              {t("fields.empty.cta")}
            </Button>
          </div>
        ) : (
          <ul className="space-y-2 p-3">
            {fields.map((f) => (
              <li key={f.id}>
                <FieldCard
                  field={f}
                  selected={selectedId === f.id}
                  onSelect={onSelect}
                  onEdit={onEdit}
                  onDelete={onDelete}
                />
              </li>
            ))}
          </ul>
        )}
      </div>
    </aside>
  );
}
