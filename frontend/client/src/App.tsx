import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import PageShell from './components/layout/PageShell';
import AdminLayout from './components/admin/AdminLayout';

import Overview from './pages/Overview';
import WalletLookup from './pages/WalletLookup';
import ClusterExplorer from './pages/ClusterExplorer';
import TransactionPath from './pages/TransactionPath';

import AdminLogin from './pages/admin/AdminLogin';
import SystemHealth from './pages/admin/SystemHealth';
import ModelRegistry from './pages/admin/ModelRegistry';
import BatchJobManager from './pages/admin/BatchJobManager';
import FeatureAudit from './pages/admin/FeatureAudit';
import LoadTestResults from './pages/admin/LoadTestResults';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<PageShell />}>
          <Route index element={<Overview />} />
          <Route path="wallet" element={<WalletLookup />} />
          <Route path="wallet/:address" element={<WalletLookup />} />
          <Route path="clusters" element={<ClusterExplorer />} />
          <Route path="path" element={<TransactionPath />} />
        </Route>
        
        <Route path="/admin/login" element={<AdminLogin />} />
        
        <Route path="/admin" element={<AdminLayout />}>
          <Route index element={<Navigate to="health" replace />} />
          <Route path="health" element={<SystemHealth />} />
          <Route path="models" element={<ModelRegistry />} />
          <Route path="jobs" element={<BatchJobManager />} />
          <Route path="features" element={<FeatureAudit />} />
          <Route path="loadtest" element={<LoadTestResults />} />
        </Route>
        
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
