import React, { useRef, useEffect, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

export default function GraphVisualizer({ nodes, edges, centerNodeId, height = 400 }) {
  const fgRef = useRef();
  const [dimensions, setDimensions] = useState({ width: 800, height });
  const containerRef = useRef(null);

  useEffect(() => {
    if (containerRef.current) {
      setDimensions({
        width: containerRef.current.offsetWidth,
        height
      });
    }
  }, [height]);

  // Format data for react-force-graph
  const graphData = {
    nodes: nodes.map(n => ({
      id: n.txId || n.id,
      name: n.txId || n.id,
      val: (n.risk_score || 0.5) * 10, // size based on risk
      color: n.txId === centerNodeId ? '#3b82f6' : (n.risk_score >= 0.7 ? '#ef4444' : '#10b981'),
      ...n
    })),
    links: edges.map(e => ({
      source: e.source || e.src,
      target: e.target || e.dst,
      color: 'rgba(255,255,255,0.2)'
    }))
  };

  useEffect(() => {
    // Zoom to fit on load
    if (fgRef.current && graphData.nodes.length > 0) {
      setTimeout(() => {
        fgRef.current.zoomToFit(400, 50);
      }, 500);
    }
  }, [graphData.nodes.length]);

  return (
    <div ref={containerRef} style={{ width: '100%', height, borderRadius: '12px', overflow: 'hidden', background: 'rgba(0,0,0,0.3)' }}>
      <ForceGraph2D
        ref={fgRef}
        width={dimensions.width}
        height={dimensions.height}
        graphData={graphData}
        nodeLabel="name"
        nodeColor="color"
        linkColor="color"
        linkDirectionalArrowLength={3.5}
        linkDirectionalArrowRelPos={1}
        nodeCanvasObjectMode={() => 'after'}
        nodeCanvasObject={(node, ctx, globalScale) => {
          const label = node.name.substring(0, 5) + '...';
          const fontSize = 12 / globalScale;
          ctx.font = `${fontSize}px Sans-Serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
          ctx.fillText(label, node.x, node.y + 8);
        }}
      />
    </div>
  );
}
