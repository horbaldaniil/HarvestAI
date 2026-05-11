import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import type { Polygon as GJPolygon } from "geojson";
import type { AxiosError } from "axios";

import { AppShell } from "@/components/layout/AppShell";
import { FieldMap } from "@/components/map/FieldMap";
import { FieldPolygon } from "@/components/map/FieldPolygon";
import { FieldFocusController } from "@/components/map/FieldFocusController";
import { PolygonDrawer, type LDrawer } from "@/components/map/PolygonDrawer";
import { FieldsSidebar } from "@/components/fields/FieldsSidebar";
import {
  FieldFormDialog,
  type FieldFormValues,
} from "@/components/fields/FieldFormDialog";
import { DeleteFieldDialog } from "@/components/fields/DeleteFieldDialog";
import {
  useCreateField,
  useDeleteField,
  useFields,
  useUpdateField,
} from "@/hooks/useFields";
import type { FieldRead } from "@/api/fields";

type FormState =
  | { open: false }
  | { open: true; mode: "create"; polygon: GJPolygon }
  | { open: true; mode: "edit"; field: FieldRead };

export function FieldsPage() {
  const { t } = useTranslation();

  const fieldsQuery = useFields();
  const createMut = useCreateField();
  const updateMut = useUpdateField();
  const deleteMut = useDeleteField();

  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [drawing, setDrawing] = useState(false);
  const [form, setForm] = useState<FormState>({ open: false });
  const [deleteCandidate, setDeleteCandidate] = useState<FieldRead | null>(null);
  // Live count of vertices placed in the in-progress polygon; gates the
  // "Finish" button so the user can't try to complete with < 3 vertices.
  const [vertexCount, setVertexCount] = useState(0);
  const drawerRef = useRef<LDrawer | null>(null);

  const fields = fieldsQuery.data ?? [];
  const selectedField = useMemo(
    () => fields.find((f) => f.id === selectedId) ?? null,
    [fields, selectedId],
  );

  // Poll the drawer's marker count while drawing. leaflet-draw does not emit
  // a vertex-added event we can hook into, so a 200ms interval is the
  // pragmatic way to keep the "Finish" button's enabled state in sync.
  useEffect(() => {
    if (!drawing) {
      setVertexCount(0);
      return;
    }
    const tick = () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const markers = (drawerRef.current as any)?._markers as unknown[] | undefined;
      setVertexCount(markers?.length ?? 0);
    };
    tick();
    const id = window.setInterval(tick, 200);
    return () => window.clearInterval(id);
  }, [drawing]);

  // ───── Handlers ──────────────────────────────────────────────

  const handleStartDraw = () => {
    setDrawing(true);
    setSelectedId(null);
  };

  const handleCancelDraw = () => {
    setDrawing(false);
  };

  const handleFinishDraw = useCallback(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const d = drawerRef.current as any;
    if (d?._markers && d._markers.length >= 3) {
      d._finishShape();
    }
  }, []);

  const handleDrawerReady = useCallback((d: LDrawer | null) => {
    drawerRef.current = d;
  }, []);

  const handlePolygonComplete = (polygon: GJPolygon) => {
    setDrawing(false);
    setForm({ open: true, mode: "create", polygon });
  };

  const handleSelect = (field: FieldRead) => {
    setSelectedId(field.id);
  };

  const handleEdit = (field: FieldRead) => {
    setForm({ open: true, mode: "edit", field });
  };

  const handleDelete = (field: FieldRead) => {
    setDeleteCandidate(field);
  };

  const handleFormClose = (open: boolean) => {
    if (!open) setForm({ open: false });
  };

  const handleFormSubmit = async (values: FieldFormValues) => {
    if (!form.open) return;
    try {
      if (form.mode === "create") {
        const created = await createMut.mutateAsync({
          name: values.name,
          crop_type: values.crop_type,
          season_year: values.season_year,
          geometry: form.polygon,
        });
        toast.success(t("fields.create.success"));
        setSelectedId(created.id);
        setForm({ open: false });
      } else {
        const updated = await updateMut.mutateAsync({
          id: form.field.id,
          payload: {
            name: values.name,
            crop_type: values.crop_type,
            season_year: values.season_year,
          },
        });
        toast.success(t("fields.edit.success"));
        setSelectedId(updated.id);
        setForm({ open: false });
      }
    } catch (err) {
      const ax = err as AxiosError<{ detail?: string }>;
      const message =
        ax.response?.data?.detail ??
        t(form.mode === "create" ? "fields.create.failed" : "fields.edit.failed");
      toast.error(message);
    }
  };

  const handleConfirmDelete = async () => {
    if (!deleteCandidate) return;
    try {
      await deleteMut.mutateAsync(deleteCandidate.id);
      toast.success(t("fields.delete.success"));
      if (selectedId === deleteCandidate.id) setSelectedId(null);
      setDeleteCandidate(null);
    } catch (err) {
      const ax = err as AxiosError<{ detail?: string }>;
      toast.error(ax.response?.data?.detail ?? t("fields.delete.failed"));
    }
  };

  // ───── Render ────────────────────────────────────────────────

  const pendingPolygon = form.open && form.mode === "create" ? form.polygon : null;
  const editingField = form.open && form.mode === "edit" ? form.field : null;

  return (
    <AppShell fullBleed>
      <div className="relative flex h-full">
        <FieldsSidebar
          fields={fields}
          isLoading={fieldsQuery.isLoading}
          selectedId={selectedId}
          drawing={drawing}
          canFinishDrawing={vertexCount >= 3}
          onStartDraw={handleStartDraw}
          onCancelDraw={handleCancelDraw}
          onFinishDraw={handleFinishDraw}
          onSelect={handleSelect}
          onEdit={handleEdit}
          onDelete={handleDelete}
        />

        <div className="relative flex-1">
          <FieldMap>
            {fields.map((f) => (
              <FieldPolygon
                key={f.id}
                field={f}
                selected={selectedId === f.id}
                onClick={handleSelect}
              />
            ))}
            <FieldFocusController field={selectedField} />
            <PolygonDrawer
              active={drawing}
              onComplete={handlePolygonComplete}
              onCancel={handleCancelDraw}
              onDrawerReady={handleDrawerReady}
            />
          </FieldMap>

          {drawing && (
            <div className="pointer-events-none absolute left-1/2 top-4 z-[1000] -translate-x-1/2 rounded-full bg-foreground px-4 py-2 text-xs font-medium text-background shadow-lg">
              {t("fields.hint.drawing")}
            </div>
          )}
        </div>

        <FieldFormDialog
          open={form.open}
          onOpenChange={handleFormClose}
          mode={form.open ? form.mode : "create"}
          pendingPolygon={pendingPolygon}
          field={editingField}
          isSubmitting={createMut.isPending || updateMut.isPending}
          onSubmit={handleFormSubmit}
        />

        <DeleteFieldDialog
          field={deleteCandidate}
          open={!!deleteCandidate}
          onOpenChange={(open) => !open && setDeleteCandidate(null)}
          isDeleting={deleteMut.isPending}
          onConfirm={handleConfirmDelete}
        />
      </div>
    </AppShell>
  );
}
