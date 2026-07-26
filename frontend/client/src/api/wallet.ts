import { useQuery } from '@tanstack/react-query';
import { apiClient } from './client';
import type { WalletInfo, SubgraphData, PathData } from '../types';

export const useWallet = (address: string) => {
  return useQuery({
    queryKey: ['wallet', address],
    queryFn: async (): Promise<WalletInfo> => {
      const { data } = await apiClient.get(`/wallet/${address}`);
      return data;
    },
    enabled: !!address && address.trim().length > 0,
    retry: 1,
  });
};

export const useSubgraph = (address: string, hops: number = 2) => {
  return useQuery({
    queryKey: ['subgraph', address, hops],
    queryFn: async (): Promise<SubgraphData> => {
      const { data } = await apiClient.get(`/wallet/${address}/subgraph`, { params: { hops } });
      return data;
    },
    enabled: !!address && address.trim().length > 0,
  });
};

export const usePath = (src: string, dst: string) => {
  return useQuery({
    queryKey: ['path', src, dst],
    queryFn: async (): Promise<PathData> => {
      const { data } = await apiClient.get('/wallet/path/find', { params: { src, dst } });
      return data;
    },
    enabled: !!src && !!dst,
  });
};
