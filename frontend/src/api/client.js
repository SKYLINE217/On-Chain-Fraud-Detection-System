const BASE_URL = '/api';

export const apiClient = {
  getWallet: async (address) => {
    const res = await fetch(`${BASE_URL}/wallet/${address}`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
  getSubgraph: async (address, hops = 2) => {
    const res = await fetch(`${BASE_URL}/wallet/${address}/subgraph?hops=${hops}`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
  getTopClusters: async (limit = 100) => {
    const res = await fetch(`${BASE_URL}/cluster/top?limit=${limit}`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
  getClusterDetails: async (clusterId) => {
    const res = await fetch(`${BASE_URL}/cluster/${clusterId}`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
  getPath: async (src, dst) => {
    const res = await fetch(`${BASE_URL}/path?src=${src}&dst=${dst}`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
  explainPrediction: async (address) => {
    const res = await fetch(`${BASE_URL}/explain/${address}`, {
      method: 'POST'
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }
};
