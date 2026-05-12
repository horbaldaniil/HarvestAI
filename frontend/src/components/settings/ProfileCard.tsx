import { useEffect, useState } from "react";
import { Loader2, Save, UserCog } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useProfile, useUpdateProfile } from "@/hooks/useProfile";

/**
 * Editable profile card — sits alongside AccountCard on /settings.
 *
 * Fields:
 *   - Повне ім'я (`full_name`, mapped to `users.full_name` column)
 *   - Назва господарства (`farm_name`, JSONB)
 *   - Телефон (`phone_number`, JSONB)
 *
 * Empty input on save = clear that field. The backend treats `""` and
 * `null` identically — see `_empty_to_none` in `app/routers/settings.py`.
 */
export function ProfileCard() {
  const { data, isLoading } = useProfile();
  const update = useUpdateProfile();

  const [fullName, setFullName] = useState("");
  const [farmName, setFarmName] = useState("");
  const [phoneNumber, setPhoneNumber] = useState("");

  useEffect(() => {
    if (!data) return;
    setFullName(data.full_name ?? "");
    setFarmName(data.farm_name ?? "");
    setPhoneNumber(data.phone_number ?? "");
  }, [data]);

  const handleSave = () => {
    update.mutate({
      full_name: fullName,
      farm_name: farmName,
      phone_number: phoneNumber,
    });
  };

  // Disable save until at least one field has changed vs server state —
  // avoids "submit identical payload" round-trips when the user just
  // looked at the form.
  const dirty =
    (data?.full_name ?? "") !== fullName ||
    (data?.farm_name ?? "") !== farmName ||
    (data?.phone_number ?? "") !== phoneNumber;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <UserCog className="h-4 w-4 text-primary" />
          Профіль фермера
        </CardTitle>
        <p className="text-xs text-muted-foreground">
          Контактна інформація та дані господарства. Використовуються
          у звітах і — в майбутньому — для сповіщень.
        </p>
      </CardHeader>
      <CardContent className="space-y-3">
        {isLoading ? (
          <div className="flex justify-center py-4">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
          </div>
        ) : (
          <>
            <Field
              id="profile-full-name"
              label="Повне ім'я"
              value={fullName}
              onChange={setFullName}
              placeholder="Іван Петренко"
              maxLength={255}
            />
            <Field
              id="profile-farm-name"
              label="Назва господарства"
              value={farmName}
              onChange={setFarmName}
              placeholder="ФГ «Лан»"
              maxLength={255}
            />
            <Field
              id="profile-phone"
              label="Телефон"
              value={phoneNumber}
              onChange={setPhoneNumber}
              placeholder="+380 50 123 45 67"
              maxLength={64}
              type="tel"
            />
            <Button
              onClick={handleSave}
              disabled={update.isPending || !dirty}
              className="w-full"
            >
              {update.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Save className="h-4 w-4" />
              )}
              Зберегти
            </Button>
          </>
        )}
      </CardContent>
    </Card>
  );
}


function Field({
  id,
  label,
  value,
  onChange,
  placeholder,
  maxLength,
  type = "text",
}: {
  id: string;
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder: string;
  maxLength: number;
  type?: "text" | "tel";
}) {
  return (
    <div>
      <Label htmlFor={id}>{label}</Label>
      <Input
        id={id}
        className="mt-1.5"
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        maxLength={maxLength}
      />
    </div>
  );
}
