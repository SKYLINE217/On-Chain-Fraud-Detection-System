import axios from 'axios';
import { useAuthStore } from '../store/authStore';

// Determine the base URL based on environment
// In dev, Vite proxies /api to http://localhost:3000
// In prod, BFF serves React and the /api route is on the same host
const baseURL = '/api';

export const apiClient = axios.create({
  baseURL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add a request interceptor to inject the token for all routes
apiClient.interceptors.request.use(
  (config) => {
    const token = useAuthStore.getState().token;
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);
