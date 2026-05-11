import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { Card } from "@/components/ui/card";
import { FAQ } from "@/lib/knowledge";
import { cn } from "@/lib/utils";

export function FaqTab() {
  const [open, setOpen] = useState<string | null>(null);

  return (
    <div className="space-y-2">
      {FAQ.map((item) => {
        const isOpen = open === item.id;
        return (
          <Card key={item.id} className="overflow-hidden">
            <button
              type="button"
              onClick={() => setOpen(isOpen ? null : item.id)}
              className={cn(
                "flex w-full items-start gap-3 p-4 text-left transition-colors hover:bg-accent/30",
                isOpen && "bg-accent/20",
              )}
            >
              {isOpen ? (
                <ChevronDown className="h-4 w-4 shrink-0 text-muted-foreground" />
              ) : (
                <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground" />
              )}
              <span className="text-sm font-medium">{item.q}</span>
            </button>
            {isOpen && (
              <div className="markdown-body border-t bg-background p-4 text-sm">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {item.a}
                </ReactMarkdown>
              </div>
            )}
          </Card>
        );
      })}
    </div>
  );
}
