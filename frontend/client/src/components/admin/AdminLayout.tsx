import { Outlet, Navigate, NavLink } from 'react-router-dom';
import { useAuthStore } from '@/store/auth';
import { Activity, Database, Server, Settings, ArrowLeft, LogOut } from 'lucide-react';

export default function AdminLayout() {
  const { token, logout } = useAuthStore();

  if (!token) {
    return <Navigate to="/admin/login" replace />;
  }

  const links = [
    { name: 'System Health', path: '/admin/health', icon: Activity },
    { name: 'Model Registry', path: '/admin/models', icon: Database },
    { name: 'Batch Jobs', path: '/admin/jobs', icon: Server },
    { name: 'Feature Audit', path: '/admin/features', icon: Settings },
    { name: 'Load Test', path: '/admin/loadtest', icon: Activity },
  ];

  return (
    <div className="min-h-screen bg-background text-foreground flex">
      <aside className="w-[240px] fixed inset-y-0 left-0 bg-surface border-r border-border flex flex-col z-10">
        <div className="p-4 border-b border-border">
          <NavLink to="/" className="flex items-center gap-2 text-sm text-muted-foreground hover:text-accent transition-colors">
            <ArrowLeft className="w-4 h-4" /> Back to Dashboard
          </NavLink>
        </div>
        
        <div className="flex-1 py-6 px-3 flex flex-col gap-2">
          <div className="px-3 mb-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            Admin Console
          </div>
          {links.map((link) => (
            <NavLink
              key={link.name}
              to={link.path}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive ? 'bg-accent/10 text-accent' : 'text-muted-foreground hover:bg-elevated hover:text-foreground'
                }`
              }
            >
              <link.icon className="w-4 h-4" />
              {link.name}
            </NavLink>
          ))}
        </div>

        <div className="p-4 border-t border-border">
          <button
            onClick={logout}
            className="flex items-center gap-2 text-sm text-muted-foreground hover:text-destructive w-full px-3 py-2 rounded-md transition-colors"
          >
            <LogOut className="w-4 h-4" /> Sign Out
          </button>
        </div>
      </aside>

      <main className="flex-1 ml-[240px] p-8 min-h-screen bg-base">
        <Outlet />
      </main>
    </div>
  );
}
