import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/layout/Layout';
import Overview from './pages/Overview';
import WalletLookup from './pages/WalletLookup';

// Placeholders for other pages
const ClusterExplorer = () => <div className="p-4">Cluster Explorer (WIP)</div>;
const TransactionPath = () => <div className="p-4">Transaction Path (WIP)</div>;
const AdminLogin = () => <div className="p-4">Admin Login (WIP)</div>;

function App() {
  return (
    <BrowserRouter>
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
    </BrowserRouter>
  );
}

export default App;
