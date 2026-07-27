import { NavLink, useLocation } from 'react-router-dom';
import { Network, ArrowRight, Menu } from 'lucide-react';
import { useAuthStore } from '@/store/auth';
import { useState } from 'react';

export default function NavBar() {
  const { isAdmin } = useAuthStore();
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const isAdminRoute = location.pathname.startsWith('/admin');

  const navLinks = [
    { name: 'Overview', path: '/' },
    { name: 'Wallet', path: '/wallet' },
    { name: 'Clusters', path: '/clusters' },
    { name: 'Path', path: '/path' },
  ];

  return (
    <nav className="sticky top-0 z-40 w-full backdrop-blur-xl bg-background/80 border-b border-border h-14">
      <div className="max-w-[1440px] mx-auto px-4 md:px-12 h-full flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Network className="w-6 h-6 text-accent" />
          <span className="font-bold tracking-tight text-foreground hidden sm:inline-block">
            On-Chain Fraud Detector
          </span>
        </div>

        {}
        <div className="hidden md:flex items-center gap-1">
          {navLinks.map((link) => (
            <NavLink
              key={link.name}
              to={link.path}
              className={({ isActive }) =>
                `px-3 py-2 rounded-md text-sm font-medium transition-colors hover:text-foreground ${
                  isActive ? 'text-accent border-b-2 border-accent rounded-b-none' : 'text-muted-foreground'
                }`
              }
            >
              {link.name}
            </NavLink>
          ))}
        </div>

        <div className="hidden md:flex items-center">
          {(!isAdmin || !isAdminRoute) && (
            <NavLink
              to="/admin/login"
              className="flex items-center gap-1 text-sm font-medium text-muted-foreground hover:text-accent transition-colors"
            >
              Admin <ArrowRight className="w-4 h-4" />
            </NavLink>
          )}
        </div>

        {}
        <button
          className="md:hidden p-2 text-muted-foreground"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
        >
          <Menu className="w-5 h-5" />
        </button>
      </div>

      {}
      {mobileMenuOpen && (
        <div className="md:hidden absolute top-14 left-0 w-full bg-surface border-b border-border p-4 flex flex-col gap-4 shadow-modal">
          {navLinks.map((link) => (
            <NavLink
              key={link.name}
              to={link.path}
              onClick={() => setMobileMenuOpen(false)}
              className={({ isActive }) =>
                `block px-3 py-2 rounded-md text-base font-medium ${
                  isActive ? 'bg-accent/10 text-accent' : 'text-muted-foreground hover:bg-elevated'
                }`
              }
            >
              {link.name}
            </NavLink>
          ))}
          {(!isAdmin || !isAdminRoute) && (
            <NavLink
              to="/admin/login"
              onClick={() => setMobileMenuOpen(false)}
              className="flex items-center gap-2 px-3 py-2 text-base font-medium text-muted-foreground hover:bg-elevated rounded-md"
            >
              Admin <ArrowRight className="w-4 h-4" />
            </NavLink>
          )}
        </div>
      )}
    </nav>
  );
}
