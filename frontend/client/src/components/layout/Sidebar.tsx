import { Link, useLocation } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { useUiStore } from '@/store/uiStore';
import { 
  Activity, 
  Search, 
  Network, 
  Route, 
  Settings, 
  ShieldAlert
} from 'lucide-react';

const navigation = [
  { name: 'Overview', href: '/', icon: Activity },
  { name: 'Wallet Lookup', href: '/wallet', icon: Search },
  { name: 'Cluster Explorer', href: '/clusters', icon: Network },
  { name: 'Transaction Path', href: '/path', icon: Route },
];

export default function Sidebar() {
  const location = useLocation();
  const { sidebarOpen, setSidebarOpen } = useUiStore();

  return (
    <>
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 z-40 bg-background/80 backdrop-blur-sm md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div className={cn(
        "fixed inset-y-0 left-0 z-50 w-64 border-r border-border bg-background transform transition-transform duration-200 ease-in-out md:translate-x-0 md:static md:block",
        sidebarOpen ? "translate-x-0" : "-translate-x-full"
      )}>
        <div className="flex h-16 shrink-0 items-center px-6 border-b border-border">
          <ShieldAlert className="h-8 w-8 text-destructive" />
          <span className="ml-3 text-lg font-semibold text-foreground">OnChain Sentinel</span>
        </div>
        
        <nav className="flex flex-1 flex-col px-4 pt-6 space-y-2">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href;
            return (
              <Link
                key={item.name}
                to={item.href}
                onClick={() => setSidebarOpen(false)}
                className={cn(
                  "group flex items-center px-3 py-2.5 text-sm font-medium rounded-md transition-colors",
                  isActive 
                    ? "bg-secondary text-secondary-foreground" 
                    : "text-muted-foreground hover:bg-secondary/50 hover:text-foreground"
                )}
              >
                <item.icon
                  className={cn(
                    "mr-3 h-5 w-5 flex-shrink-0 transition-colors",
                    isActive ? "text-foreground" : "text-muted-foreground group-hover:text-foreground"
                  )}
                  aria-hidden="true"
                />
                {item.name}
              </Link>
            );
          })}
        </nav>

        <div className="p-4 border-t border-border">
          <Link
            to="/admin/login"
            className="flex items-center px-3 py-2 text-sm font-medium text-muted-foreground rounded-md hover:bg-secondary/50 hover:text-foreground transition-colors"
          >
            <Settings className="mr-3 h-5 w-5" />
            Admin Panel
          </Link>
        </div>
      </div>
    </>
  );
}
