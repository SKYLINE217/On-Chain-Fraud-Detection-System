// BUG-17 Fix: Use encodeURIComponent and URLSearchParams to properly encode
// special characters in transaction IDs or addresses. Without encoding, IDs
// containing special chars would break the URL or produce 422 validation errors.

const BASE_URL = '/api';

/**
 * Build a URL with safely encoded query parameters.
 */
function buildUrl(path, params = {}) {
  const url = new URL(path, window.location.origin);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      url.searchParams.set(key, String(value));
    }
  });
  return url.pathname + url.search;
}

export const apiClient = {
  getWallet: async (address) => {
    // BUG-17: encode address in path segment
    const res = await fetch(`${BASE_URL}/wallet/${encodeURIComponent(address)}`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
  getSubgraph: async (address, hops = 2) => {
    // BUG-17: encode address in path segment; hops as numeric query param
    const url = buildUrl(`${BASE_URL}/wallet/${encodeURIComponent(address)}/subgraph`, { hops });
    const res = await fetch(url);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
  getTopClusters: async (limit = 100) => {
    const url = buildUrl(`${BASE_URL}/cluster/top`, { limit });
    const res = await fetch(url);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
  getClusterDetails: async (clusterId) => {
    const res = await fetch(`${BASE_URL}/cluster/${encodeURIComponent(clusterId)}`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
  getPath: async (src, dst) => {
    // BUG-17: Use URLSearchParams to safely encode src and dst
    const url = buildUrl(`${BASE_URL}/path/`, { src, dst });
    const res = await fetch(url);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
  explainPrediction: async (address) => {
    // BUG-17: encode address in path segment
    const res = await fetch(`${BASE_URL}/explain/${encodeURIComponent(address)}`, {
      method: 'POST'
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }
};
