import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";

interface SuggestedPromptsProps {
  onPick: (prompt: string) => void;
  disabled?: boolean;
}

/**
 * Shown in an empty session, before the user has sent a message. Each chip
 * is just shorthand for `onPick(prompt) → setInput(prompt) → send()`.
 */
export function SuggestedPrompts({ onPick, disabled }: SuggestedPromptsProps) {
  const { t } = useTranslation();
  const PROMPTS: Array<keyof typeof PROMPT_TEXT> = [
    "ndviDrop",
    "prediction",
    "recommendations",
    "risks14d",
  ];

  return (
    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
      {PROMPTS.map((k) => (
        <Button
          key={k}
          variant="outline"
          size="sm"
          className="h-auto whitespace-normal py-2 text-left text-xs leading-snug"
          onClick={() => onPick(t(`chat.suggested.${k}`))}
          disabled={disabled}
        >
          {t(`chat.suggested.${k}`)}
        </Button>
      ))}
    </div>
  );
}

// Used only for the type narrowing — values come from i18n.
const PROMPT_TEXT = {
  ndviDrop: "",
  prediction: "",
  recommendations: "",
  risks14d: "",
} as const;
