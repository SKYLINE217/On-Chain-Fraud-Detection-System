import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Network, AlertCircle } from 'lucide-react';
import { useAuthStore } from '@/store/auth';

export default function AdminLogin() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const setToken = useAuthStore(s => s.setToken);
  const navigate = useNavigate();

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    if (username === 'admin' && password === 'admin') {
      setToken('mock-jwt-token');
      navigate('/admin/health');
    } else {
      setError('Invalid credentials');
    }
  };

  return (
    <div className="min-h-screen bg-[var(--color-bg-base)] flex flex-col items-center justify-center p-4">
      <div className="bg-[var(--color-bg-surface)] border border-[var(--color-border)] p-8 rounded-lg shadow-modal w-full max-w-[400px]">
        <div className="flex flex-col items-center mb-8">
          <Network className="w-12 h-12 text-[var(--color-accent)] mb-4" />
          <h1 className="text-xl font-bold text-[var(--color-text-primary)]">On-Chain Fraud Detector</h1>
          <p className="text-[var(--color-text-secondary)] text-sm mt-1">Developer Admin Access</p>
        </div>

        <form onSubmit={handleLogin} className="flex flex-col gap-4">
          <div className="flex flex-col gap-1">
            <label className="text-sm text-[var(--color-text-secondary)]">Username</label>
            <input 
              type="text" 
              value={username}
              onChange={e => setUsername(e.target.value)}
              className="bg-[var(--color-bg-sunken)] border border-[var(--color-border)] rounded px-3 py-2 text-[var(--color-text-primary)] focus:border-[var(--color-accent)] outline-none"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-sm text-[var(--color-text-secondary)]">Password</label>
            <input 
              type="password" 
              value={password}
              onChange={e => setPassword(e.target.value)}
              className="bg-[var(--color-bg-sunken)] border border-[var(--color-border)] rounded px-3 py-2 text-[var(--color-text-primary)] focus:border-[var(--color-accent)] outline-none"
            />
          </div>

          {error && (
            <div className="flex items-center gap-2 text-[var(--color-text-danger)] text-sm mt-2">
              <AlertCircle className="w-4 h-4" /> {error}
            </div>
          )}

          <button type="submit" className="bg-[var(--color-accent)] text-white py-2 rounded font-medium hover:bg-[var(--color-accent)]/80 transition-colors mt-4">
            Sign in
          </button>
        </form>
      </div>
    </div>
  );
}
