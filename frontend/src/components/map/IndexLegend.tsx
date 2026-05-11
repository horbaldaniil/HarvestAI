import { useTranslation } from "react-i18next";
import type { IndexName } from "@/api/observations";

interface IndexLegendProps {
  index: IndexName;
  date: string;
}

/** Colour stops match the ramp() function in backend evalscripts. */
const GRADIENT = "linear-gradient(to right, #a50f15, #de2d26, #fb9a29, #ffff66, #addd8e, #31a354)";

export function IndexLegend({ index, date }: IndexLegendProps) {
  const { t } = useTranslation();
  return (
    <div className="pointer-events-auto absolute bottom-4 left-4 z-[800] rounded-md border bg-card/95 p-3 shadow-md backdrop-blur-sm">
      <div className="mb-1 text-xs font-semibold">
        {t(`indices.${index}`)} · {date}
      </div>
      <div className="h-2 w-48 rounded-sm" style={{ background: GRADIENT }} />
      <div className="mt-1 flex justify-between text-[10px] text-muted-foreground">
        <span>низькa</span>
        <span>середня</span>
        <span>висока</span>
      </div>
    </div>
  );
}
