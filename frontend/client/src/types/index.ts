export interface WalletInfo {
  address: string;
  risk_score: number;
  predicted_label: 'illicit' | 'licit' | 'unknown';
  confidence: number;
  timeStep: number;
  communityId: number;
}

export interface GraphNode {
  id: string;
  risk_score: number;
  predicted_label: string;
  communityId: number;
  timeStep: number;
}

export interface GraphEdge {
  src: string;
  dst: string;
}

export interface SubgraphData {
  address: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  hops: number;
  node_count: number;
}

export interface PathData {
  src: string;
  dst: string;
  found: boolean;
  path_nodes: GraphNode[];
  hops: number | null;
}
