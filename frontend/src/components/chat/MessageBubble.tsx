import { Wrench } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { cn } from "@/lib/utils";
import type { ChatMessageRead } from "@/api/chat";

interface MessageBubbleProps {
  message: ChatMessageRead;
}

/**
 * One row in the chat panel.
 *
 * - user      → primary-coloured bubble, right-aligned
 * - assistant → muted bubble, left-aligned, markdown-rendered
 * - tool      → skipped (the assistant turn that follows references the
 *               result; we show a single inline chip per call instead)
 */
export function MessageBubble({ message }: MessageBubbleProps) {
  if (message.role === "tool" || message.role === "system") return null;

  const isUser = message.role === "user";
  return (
    <div className={cn("flex w-full", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[88%] rounded-lg px-3 py-2 text-sm leading-relaxed shadow-sm",
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-muted text-foreground",
        )}
      >
        {message.tool_calls && message.tool_calls.length > 0 && (
          <div className="mb-2 space-y-1">
            {message.tool_calls.map((tc, i) => (
              <ToolCallChip key={`${message.id}-${i}`} name={tc.name} />
            ))}
          </div>
        )}
        {isUser ? (
          <div className="whitespace-pre-wrap break-words">{message.content}</div>
        ) : (
          <div className="markdown-body break-words">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content || ""}
            </ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  );
}

export function ToolCallChip({ name }: { name: string }) {
  return (
    <div className="inline-flex items-center gap-1.5 rounded bg-background/60 px-2 py-0.5 text-[10px] font-mono uppercase tracking-wide text-muted-foreground">
      <Wrench className="h-3 w-3" />
      {name}
    </div>
  );
}

export function StreamingBubble({
  content,
  toolNames,
}: {
  content: string;
  toolNames: string[];
}) {
  return (
    <div className="flex w-full justify-start">
      <div className="max-w-[88%] rounded-lg bg-muted px-3 py-2 text-sm shadow-sm">
        {toolNames.length > 0 && (
          <div className="mb-2 space-y-1">
            {toolNames.map((name, i) => (
              <ToolCallChip key={i} name={name} />
            ))}
          </div>
        )}
        <div className="markdown-body break-words">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {content || "…"}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
}
