import { create } from 'zustand';

interface AuthState {
  token: string | null;
  isAdmin: boolean;
  login: (token: string) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>()((set) => ({
  token: null,
  isAdmin: false,
  login: (token) => set({ token, isAdmin: true }),
  logout: () => set({ token: null, isAdmin: false }),
}));
