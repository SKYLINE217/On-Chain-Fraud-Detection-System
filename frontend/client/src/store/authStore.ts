import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface AuthState {
  token: string | null;
  isAdmin: boolean;
  login: (token: string) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      token: null,
      isAdmin: false,
      login: (token) => set({ token, isAdmin: true }),
      logout: () => set({ token: null, isAdmin: false }),
    }),
    {
      name: 'auth-storage',
    }
  )
);
