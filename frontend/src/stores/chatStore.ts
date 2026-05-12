import { create } from "zustand";

interface ChatStore {
  /** Whether the floating chat panel is visible. */
  isOpen: boolean;
  /** The session currently open in the panel (null = no session selected). */
  activeSessionId: number | null;
  /**
   * Field-id context for when the user opens the chat. Used to scope a new
   * session to a field when the user clicks the chat button from a field page.
   */
  contextFieldId: number | null;
  /**
   * Auto-send queue. When set, ChatPanel will create a fresh session (if
   * none is active), send this prompt, then clear the field. Used by the
   * "Templates" tab in /chat: one click → new session → answer streaming.
   */
  pendingPrompt: string | null;
  /**
   * Pixels reserved by another fixed-bottom UI element (BottomPanel on
   * /fields, future docked modals, etc.). The FloatingChatButton lifts
   * itself above this so it doesn't sit on top of in-page controls.
   */
  bottomReserveHeight: number;
  open: () => void;
  close: () => void;
  toggle: () => void;
  setActiveSession: (id: number | null) => void;
  setContextField: (id: number | null) => void;
  setBottomReserveHeight: (h: number) => void;
  /**
   * Open the panel, start a new session, and auto-send `prompt`. Existing
   * conversation is preserved — we just create a new session alongside.
   */
  openWithPrompt: (prompt: string) => void;
  clearPendingPrompt: () => void;
}

export const useChatStore = create<ChatStore>((set) => ({
  isOpen: false,
  activeSessionId: null,
  contextFieldId: null,
  pendingPrompt: null,
  bottomReserveHeight: 0,
  open: () => set({ isOpen: true }),
  close: () => set({ isOpen: false }),
  toggle: () => set((s) => ({ isOpen: !s.isOpen })),
  setActiveSession: (id) => set({ activeSessionId: id }),
  setContextField: (id) => set({ contextFieldId: id }),
  setBottomReserveHeight: (h) => set({ bottomReserveHeight: Math.max(0, h) }),
  openWithPrompt: (prompt) =>
    set({
      isOpen: true,
      // Force a new session so the template starts fresh, not appended to an
      // unrelated prior conversation.
      activeSessionId: null,
      pendingPrompt: prompt,
    }),
  clearPendingPrompt: () => set({ pendingPrompt: null }),
}));
