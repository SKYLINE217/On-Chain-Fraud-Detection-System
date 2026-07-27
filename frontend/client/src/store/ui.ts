import { create } from 'zustand';

interface UiState {
  labelsVisible: boolean;
  toggleLabels: () => void;
}

export const useUiStore = create<UiState>((set) => ({
  labelsVisible: true,
  toggleLabels: () => set((state) => ({ labelsVisible: !state.labelsVisible })),
}));
