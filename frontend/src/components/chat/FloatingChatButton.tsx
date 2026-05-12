import { MessageCircle } from "lucide-react";
import { useTranslation } from "react-i18next";

import { cn } from "@/lib/utils";
import { useChatStore } from "@/stores/chatStore";

// Default base offset matches what `bottom-6` would give (1.5rem = 24px).
const BASE_BOTTOM_PX = 24;
// Visual breathing room between the button and whatever is reserving the
// bottom of the viewport (e.g. BottomPanel on /fields).
const GAP_ABOVE_PANEL_PX = 16;

/**
 * Floating action button anchored to the bottom-right of the viewport.
 * Lives in AppShell so it appears on every page. Z-index 1900 sits above
 * Leaflet panes (~800) but below Radix Dialog overlays (2000+).
 *
 * `bottomReserveHeight` from the chat store lets pages with their own
 * docked bottom UI (BottomPanel on /fields, future modals) push this
 * button up so it never sits on top of in-page controls.
 */
export function FloatingChatButton() {
  const { t } = useTranslation();
  const isOpen = useChatStore((s) => s.isOpen);
  const toggle = useChatStore((s) => s.toggle);
  const reserve = useChatStore((s) => s.bottomReserveHeight);

  const bottomPx = reserve > 0 ? BASE_BOTTOM_PX + reserve + GAP_ABOVE_PANEL_PX : BASE_BOTTOM_PX;

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={t("chat.openLabel")}
      style={{ bottom: `${bottomPx}px` }}
      className={cn(
        "fixed right-6 z-[1900] flex h-14 w-14 items-center justify-center",
        "rounded-full bg-primary text-primary-foreground shadow-lg",
        "transition-all duration-200 hover:scale-105 hover:shadow-xl",
        "focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
        isOpen && "opacity-0 pointer-events-none",
      )}
    >
      <MessageCircle className="h-6 w-6" />
    </button>
  );
}
