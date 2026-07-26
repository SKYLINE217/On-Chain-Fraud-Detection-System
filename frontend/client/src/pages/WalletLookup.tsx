import { useState } from 'react';
import { useWallet, useSubgraph } from '@/api/wallet';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Search, ShieldAlert, ShieldCheck, HelpCircle, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

export default function WalletLookup() {
  const [searchInput, setSearchInput] = useState('');
  const [activeAddress, setActiveAddress] = useState('');

  const { data: wallet, isLoading: walletLoading, error: walletError } = useWallet(activeAddress);
  const { data: subgraph, isLoading: subgraphLoading } = useSubgraph(activeAddress, 2);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchInput.trim()) {
      setActiveAddress(searchInput.trim());
    }
  };

  const getRiskIcon = (label?: string) => {
    if (label === 'illicit') return <ShieldAlert className="h-12 w-12 text-destructive mb-2" />;
    if (label === 'licit') return <ShieldCheck className="h-12 w-12 text-green-500 mb-2" />;
    return <HelpCircle className="h-12 w-12 text-muted-foreground mb-2" />;
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Wallet Investigation</h1>
        <p className="text-muted-foreground mt-2">
          Search for an address to view its risk score, predicted label, and local transaction subgraph.
        </p>
      </div>

      <Card className="border-primary/20 bg-primary/5">
        <CardContent className="pt-6">
          <form onSubmit={handleSearch} className="flex gap-2">
            <Input
              placeholder="Enter Wallet Address (e.g. 11)"
              value={searchInput}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearchInput(e.target.value)}
              className="flex-1 font-mono bg-background"
            />
            <Button type="submit" disabled={!searchInput.trim() || walletLoading}>
              {walletLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Search className="mr-2 h-4 w-4" />}
              Analyze
            </Button>
          </form>
        </CardContent>
      </Card>

      {walletError && (
        <Card className="border-destructive/50 bg-destructive/10">
          <CardContent className="pt-6 text-destructive">
            Wallet not found or error occurred during analysis.
          </CardContent>
        </Card>
      )}

      {wallet && (
        <div className="grid gap-6 md:grid-cols-3">
          <Card className="md:col-span-1">
            <CardHeader>
              <CardTitle>Risk Assessment</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col items-center justify-center py-6 text-center">
              {getRiskIcon(wallet.predicted_label)}
              <h2 className="text-3xl font-bold font-mono mt-2 capitalize">
                {wallet.predicted_label}
              </h2>
              
              <div className="mt-6 w-full space-y-4">
                <div className="flex justify-between items-center text-sm">
                  <span className="text-muted-foreground">Risk Score</span>
                  <span className="font-mono font-bold text-lg">{(wallet.risk_score * 100).toFixed(1)}%</span>
                </div>
                <div className="h-2 bg-secondary rounded-full overflow-hidden">
                  <div 
                    className={cn(
                      "h-full transition-all duration-1000",
                      wallet.predicted_label === 'illicit' ? "bg-destructive" : "bg-green-500"
                    )} 
                    style={{ width: `${wallet.risk_score * 100}%` }} 
                  />
                </div>
                
                <div className="pt-4 border-t border-border flex justify-between items-center text-sm">
                  <span className="text-muted-foreground">Confidence</span>
                  <Badge variant="outline" className="font-mono">
                    {(wallet.confidence * 100).toFixed(1)}%
                  </Badge>
                </div>
                <div className="flex justify-between items-center text-sm">
                  <span className="text-muted-foreground">Community ID</span>
                  <Badge variant="secondary" className="font-mono">
                    {wallet.communityId}
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="md:col-span-2 flex flex-col h-[500px]">
            <CardHeader>
              <CardTitle>Local Transaction Subgraph (2-Hop)</CardTitle>
              <CardDescription>
                Visualization requires Sigma.js implementation (WIP).
                {subgraph && ` Found ${subgraph.node_count} connected entities.`}
              </CardDescription>
            </CardHeader>
            <CardContent className="flex-1 flex items-center justify-center bg-secondary/20 rounded-md m-4 border border-border border-dashed">
              {subgraphLoading ? (
                <div className="flex flex-col items-center text-muted-foreground">
                  <Loader2 className="h-8 w-8 animate-spin mb-4" />
                  Extracting graph topology...
                </div>
              ) : subgraph ? (
                <div className="text-center text-muted-foreground">
                  <NetworkIcon className="h-12 w-12 mx-auto mb-4 opacity-20" />
                  <p>Graph visualization placeholder.</p>
                  <p className="text-xs mt-2 font-mono">Nodes: {subgraph.nodes.length} | Edges: {subgraph.edges.length}</p>
                </div>
              ) : (
                <span className="text-muted-foreground text-sm">No subgraph loaded.</span>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
import { Network as NetworkIcon } from 'lucide-react';
