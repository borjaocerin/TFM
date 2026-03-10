import { create } from "zustand";

import type { Locale } from "../i18n";

type AppState = {
  locale: Locale;
  toast: string | null;
  modelMetadata: Record<string, unknown> | null;
  setLocale: (locale: Locale) => void;
  setToast: (message: string | null) => void;
  setModelMetadata: (metadata: Record<string, unknown> | null) => void;
};

export const useAppStore = create<AppState>((set) => ({
  locale: "es",
  toast: null,
  modelMetadata: null,
  setLocale: (locale) => set({ locale }),
  setToast: (toast) => set({ toast }),
  setModelMetadata: (modelMetadata) => set({ modelMetadata })
}));
