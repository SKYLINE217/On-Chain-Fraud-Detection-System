import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Suspense, lazy } from 'react';
import { ErrorBoundary } from 'react-error-boundary';
import Layout from './components/layout/Layout';

// FE-05: Add React Suspense fallback for lazy loaded components
const Overview = lazy(() => import('./pages/Overview'));
const WalletLookup = lazy(() => import('./pages/WalletLookup'));

// Placeholders for other pages
const ClusterExplorer = lazy(() => Promise.resolve({ default: () => <div className="p-4">Cluster Explorer (WIP)</div> }));
const TransactionPath = lazy(() => Promise.resolve({ default: () => <div className="p-4">Transaction Path (WIP)</div> }));
const AdminLogin = lazy(() => Promise.resolve({ default: () => <div className="p-4">Admin Login (WIP)</div> }));

function ErrorFallback({ error }: { error: Error }) {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4 text-center">
      <h2 className="text-2xl font-bold text-destructive mb-2">Something went wrong</h2>
      <p className="text-muted-foreground mb-4">{error.message}</p>
      <button 
        onClick={() => window.location.reload()} 
        className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
      >
        Reload Page
      </button>
    </div>
  );
}
const ClusterExplorer = () => <div className="p-4">Cluster Explorer (WIP)</div>;
const TransactionPath = () => <div className="p-4">Transaction Path (WIP)</div>;
const AdminLogin = () => <div className="p-4">Admin Login (WIP)</div>;

function App() {
  return (
    <ErrorBoundary FallbackComponent={ErrorFallback}>
      <BrowserRouter>
        <Suspense fallback={<div className="flex items-center justify-center min-h-screen">Loading application...</div>}>
          <Routes>
            <Route path="/" element={<Layout />}>
              <Route index element={<Overview />} />
              <Route path="wallet" element={<WalletLookup />} />
              <Route path="clusters" element={<ClusterExplorer />} />
              <Route path="path" element={<TransactionPath />} />
            </Route>
            <Route path="/admin/login" element={<AdminLogin />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Suspense>
      </BrowserRouter>
    </ErrorBoundary>
  );
}

export default App;
