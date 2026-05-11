import { useTranslation } from "react-i18next";
import { AppShell } from "@/components/layout/AppShell";
import { AccountCard } from "@/components/settings/AccountCard";
import { QuotaCard } from "@/components/settings/QuotaCard";

export function SettingsPage() {
  const { t } = useTranslation();
  return (
    <AppShell>
      <div className="space-y-6">
        <header>
          <h1 className="text-3xl font-bold tracking-tight">{t("settings.title")}</h1>
        </header>
        <div className="grid gap-4 md:grid-cols-2">
          <AccountCard />
          <QuotaCard />
        </div>
      </div>
    </AppShell>
  );
}
