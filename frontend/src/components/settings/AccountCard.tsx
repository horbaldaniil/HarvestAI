import { useTranslation } from "react-i18next";
import { User } from "lucide-react";

import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useCurrentUser } from "@/hooks/useAuth";

export function AccountCard() {
  const { t } = useTranslation();
  const { data: user } = useCurrentUser();

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <User className="h-4 w-4 text-primary" />
          {t("settings.account")}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        <div>
          <div className="text-xs text-muted-foreground">{t("settings.fullName")}</div>
          <div className="font-medium">{user?.full_name || "—"}</div>
        </div>
        <div>
          <div className="text-xs text-muted-foreground">{t("settings.email")}</div>
          <div className="font-medium">{user?.email}</div>
        </div>
      </CardContent>
    </Card>
  );
}
