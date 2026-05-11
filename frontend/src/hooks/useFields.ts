import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as fieldsApi from "@/api/fields";
import type { FieldCreatePayload, FieldUpdatePayload } from "@/api/fields";

const FIELDS_KEY = ["fields"] as const;
const fieldKey = (id: number) => ["fields", id] as const;

export function useFields() {
  return useQuery({
    queryKey: FIELDS_KEY,
    queryFn: fieldsApi.listFields,
  });
}

export function useField(id: number | null | undefined) {
  return useQuery({
    queryKey: id ? fieldKey(id) : ["fields", "disabled"],
    queryFn: () => fieldsApi.getField(id!),
    enabled: typeof id === "number",
  });
}

export function useCreateField() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (payload: FieldCreatePayload) => fieldsApi.createField(payload),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: FIELDS_KEY });
    },
  });
}

export function useUpdateField() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, payload }: { id: number; payload: FieldUpdatePayload }) =>
      fieldsApi.updateField(id, payload),
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: FIELDS_KEY });
      qc.setQueryData(fieldKey(data.id), data);
    },
  });
}

export function useDeleteField() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => fieldsApi.deleteField(id),
    onSuccess: (_data, id) => {
      qc.invalidateQueries({ queryKey: FIELDS_KEY });
      qc.removeQueries({ queryKey: fieldKey(id) });
    },
  });
}
