import { MessageCircle } from "lucide-react";
import { useTranslation } from "react-i18next";

import { cn } from "@/lib/utils";
import { useChatStore } from "@/stores/chatStore";

/**
 * Floating action button anchored to the bottom-right of the viewport.
 * Lives in AppShell so it appears on every page. Z-index 1900 sits above
 * Leaflet panes (~800) but below Radix Dialog overlays (2000+).
 */
export function FloatingChatButton() {
  const { t } = useTranslation();
  const { isOpen, toggle } = useChatStore();

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={t("chat.openLabel")}
      className={cn(
        "fixed bottom-6 right-6 z-[1900] flex h-14 w-14 items-center justify-center",
        "rounded-full bg-primary text-primary-foreground shadow-lg",
        "transition-all hover:scale-105 hover:shadow-xl",
        "focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
        isOpen && "opacity-0 pointer-events-none",
      )}
    >
      <MessageCircle className="h-6 w-6" />
    </button>
  );
}
