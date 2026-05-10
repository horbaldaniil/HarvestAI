import { Wheat } from "lucide-react";
import { cn } from "@/lib/utils";

interface LogoProps {
  className?: string;
  showText?: boolean;
}

export function Logo({ className, showText = true }: LogoProps) {
  return (
    <div className={cn("flex items-center gap-2", className)}>
      <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary text-primary-foreground shadow-sm">
        <Wheat className="h-5 w-5" />
      </div>
      {showText && (
        <span className="text-xl font-bold tracking-tight text-foreground">
          HarvestAI
        </span>
      )}
    </div>
  );
}
