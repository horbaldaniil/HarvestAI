import { useTranslation } from "react-i18next";
import { Wheat } from "lucide-react";

import type { CropType } from "@/api/dashboard";
import { ALL_CROPS } from "@/api/fields";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface Props {
  selectedCrops: CropType[];
  onCropsChange: (crops: CropType[]) => void;
}

/**
 * Compact filter bar above the dashboard. Only crop filter remains —
 * the previous time-range select was confusing because most numbers
 * (predictions, alert counts) are point-in-time and the rest use a
 * fixed 14-day window (mirroring the anomaly detector).
 */
export function FilterBar({ selectedCrops, onCropsChange }: Props) {
  const { t } = useTranslation();

  const toggleCrop = (crop: CropType) => {
    if (selectedCrops.includes(crop)) {
      onCropsChange(selectedCrops.filter((c) => c !== crop));
    } else {
      onCropsChange([...selectedCrops, crop]);
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-3 rounded-lg border bg-card p-2 text-sm">
      <div className="flex items-center gap-2">
        <Wheat className="h-4 w-4 text-muted-foreground" />
        <span className="text-xs text-muted-foreground">
          {t("dashboardFilters.crops")}:
        </span>
        <div className="flex flex-wrap gap-1">
          {ALL_CROPS.map((crop) => {
            const active = selectedCrops.includes(crop);
            return (
              <Button
                key={crop}
                size="sm"
                variant={active ? "default" : "outline"}
                onClick={() => toggleCrop(crop)}
                className={cn("h-8 px-2 text-xs", !active && "border-dashed")}
              >
                {t(`fields.crops.${crop}`)}
              </Button>
            );
          })}
          {selectedCrops.length > 0 && (
            <Button
              size="sm"
              variant="ghost"
              onClick={() => onCropsChange([])}
              className="h-8 px-2 text-xs text-muted-foreground"
            >
              {t("common.clear")}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
