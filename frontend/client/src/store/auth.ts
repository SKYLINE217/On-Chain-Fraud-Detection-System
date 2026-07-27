import { create } from 'zustand';

interface AuthState {
  token: string | null;
  isAdmin: boolean;
  setToken: (token: string | null) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  token: null,
  isAdmin: false,
  setToken: (token) => set({ token, isAdmin: !!token }),
  logout: () => set({ token: null, isAdmin: false }),
}));
