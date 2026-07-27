import { useEffect } from 'react';
import { SigmaContainer, useLoadGraph, useRegisterEvents, useCamera } from '@react-sigma/core';
import Graph from 'graphology';
import forceAtlas2 from 'graphology-layout-forceatlas2';
import { ZoomIn, ZoomOut, Maximize, Type } from 'lucide-react';
import { useUiStore } from '@/store/ui';

function GraphEvents({ onNodeClick }: { onNodeClick?: (nodeId: string) => void }) {
  const registerEvents = useRegisterEvents();
  
  useEffect(() => {
    registerEvents({
      clickNode: (event) => onNodeClick?.(event.node),
    });
  }, [registerEvents, onNodeClick]);
  
  return null;
}

function GraphLoader({ graphData }: { graphData: any }) {
  const loadGraph = useLoadGraph();
  
  useEffect(() => {
    if (!graphData || !graphData.nodes) return;
    
    const graph = new Graph();
    
    graphData.nodes.forEach((n: any) => {
      if (!graph.hasNode(n.id)) {
        graph.addNode(n.id, { 
          x: Math.random() * 100, 
          y: Math.random() * 100, 
          size: n.size || 5,
          color: n.color || '#64748b',
          label: n.label || n.id.slice(0, 8),
          borderColor: n.borderColor,
          borderSize: n.borderSize
        });
      }
    });
    
    if (graphData.edges) {
      graphData.edges.forEach((e: any) => {
        if (graph.hasNode(e.src) && graph.hasNode(e.dst)) {
          if (!graph.hasEdge(e.src, e.dst)) {
            graph.addEdge(e.src, e.dst, {
              size: e.size || 1,
              color: e.color || '#2A3350',
              type: e.type || 'arrow'
            });
          }
        }
      });
    }

    if (graph.order > 0) {
      forceAtlas2.assign(graph, { iterations: 100, settings: { adjustSizes: true } });
      loadGraph(graph);
    }
  }, [graphData, loadGraph]);
  
  return null;
}

function GraphControls() {
  const camera = useCamera();
  const { labelsVisible, toggleLabels } = useUiStore();

  return (
    <div className="absolute top-2 right-2 flex flex-col gap-1 bg-surface border border-border rounded-md shadow-card p-1 z-10">
      <button onClick={() => camera.zoomIn()} className="p-1.5 text-muted hover:text-text-primary hover:bg-elevated rounded transition-colors" title="Zoom In">
        <ZoomIn className="w-4 h-4" />
      </button>
      <button onClick={() => camera.zoomOut()} className="p-1.5 text-muted hover:text-text-primary hover:bg-elevated rounded transition-colors" title="Zoom Out">
        <ZoomOut className="w-4 h-4" />
      </button>
      <button onClick={() => camera.reset()} className="p-1.5 text-muted hover:text-text-primary hover:bg-elevated rounded transition-colors" title="Reset View">
        <Maximize className="w-4 h-4" />
      </button>
      <button onClick={toggleLabels} className={`p-1.5 rounded transition-colors ${labelsVisible ? 'text-accent bg-accent/10' : 'text-muted hover:text-text-primary hover:bg-elevated'}`} title="Toggle Labels">
        <Type className="w-4 h-4" />
      </button>
    </div>
  );
}

export function GraphCanvas({ graphData, onNodeClick, height = "100%" }: { graphData: any, onNodeClick?: (id: string) => void, height?: string | number }) {
  const { labelsVisible } = useUiStore();
  
  return (
    <div className="relative w-full border border-border rounded-md overflow-hidden bg-[var(--color-bg-sunken)]" style={{ height }}>
      <SigmaContainer style={{ width: "100%", height: "100%" }} settings={{ allowInvalidContainer: true, renderLabels: labelsVisible, defaultNodeType: 'circle', defaultEdgeType: 'arrow' }}>
        <GraphLoader graphData={graphData} />
        <GraphEvents onNodeClick={onNodeClick} />
        <GraphControls />
      </SigmaContainer>
    </div>
  );
}
