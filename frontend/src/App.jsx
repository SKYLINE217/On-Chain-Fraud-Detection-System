import { BrowserRouter as Router, Routes, Route, NavLink, useNavigate } from 'react-router-dom';
import { Shield, LayoutDashboard, Share2, Search, Map } from 'lucide-react';
import { useState } from 'react';
import './App.css';

// Lazy loading pages could be added here, but for simplicity we'll import directly
import Overview from './pages/Overview';
import Clusters from './pages/Clusters';
import WalletLookup from './pages/WalletLookup';
import PathFinder from './pages/PathFinder';

function Sidebar() {
  return (
    <div className="sidebar">
      <div className="sidebar-brand">
        <Shield size={28} />
        <span>OnChainFraud</span>
      </div>
      
      <div className="nav-links">
        <NavLink to="/" className={({isActive}) => isActive ? "nav-item active" : "nav-item"}>
          <LayoutDashboard size={20} />
          Overview
        </NavLink>
        <NavLink to="/clusters" className={({isActive}) => isActive ? "nav-item active" : "nav-item"}>
          <Share2 size={20} />
          Clusters
        </NavLink>
        <NavLink to="/wallet" className={({isActive}) => isActive ? "nav-item active" : "nav-item"}>
          <Search size={20} />
          Wallet Lookup
        </NavLink>
        <NavLink to="/paths" className={({isActive}) => isActive ? "nav-item active" : "nav-item"}>
          <Map size={20} />
          Path Finder
        </NavLink>
      </div>
    </div>
  );
}

function GlobalSearch() {
  const [query, setQuery] = useState('');
  const navigate = useNavigate();

  const handleSearch = (e) => {
    e.preventDefault();
    if (query.trim()) {
      navigate(`/wallet?address=${encodeURIComponent(query.trim())}`);
      setQuery('');
    }
  };

  return (
    <form className="search-container" onSubmit={handleSearch}>
      <Search className="search-icon" />
      <input 
        type="text" 
        className="search-input" 
        placeholder="Search Wallet Address or TxID..." 
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
    </form>
  );
}

function App() {
  return (
    <Router>
      <div className="app-container">
        <Sidebar />
        
        <main className="main-content">
          <div className="page-header">
            <div>
              {/* Optional breadcrumbs or dynamic title can go here */}
            </div>
            <GlobalSearch />
          </div>
          
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/clusters" element={<Clusters />} />
            <Route path="/wallet" element={<WalletLookup />} />
            <Route path="/paths" element={<PathFinder />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
