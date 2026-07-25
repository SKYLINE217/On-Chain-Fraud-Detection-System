document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const tabs = document.querySelectorAll('.tab-btn');
    const panes = document.querySelectorAll('.tab-pane');
    
    // Ego tab
    const egoForm = document.getElementById('search-form');
    const egoInput = document.getElementById('tx-id');
    const resultsWidget = document.getElementById('results-widget');
    
    // Path tab
    const pathForm = document.getElementById('path-form');
    const srcInput = document.getElementById('src-tx');
    const dstInput = document.getElementById('dst-tx');
    const pathResults = document.getElementById('path-results');
    
    // Cluster tab
    const btnLoadClusters = document.getElementById('btn-load-clusters');
    const clusterResults = document.getElementById('cluster-results');

    // Graph Area
    const placeholder = document.getElementById('graph-placeholder');
    const loading = document.getElementById('graph-loading');
    const graphContainer = document.getElementById('graph-container');
    const graphTitle = document.getElementById('graph-title');
    
    // D3 Setup
    const width = graphContainer.clientWidth;
    const height = graphContainer.clientHeight;
    
    const svg = d3.select('#graph-container')
        .append('svg')
        .attr('width', '100%')
        .attr('height', '100%')
        .style('display', 'none');
        
    const g = svg.append('g');
    svg.call(d3.zoom().on('zoom', (event) => {
        g.attr('transform', event.transform);
    }));

    const tooltip = d3.select('body').append('div').attr('class', 'd3-tooltip');
    let simulation = null;

    // --- Tab Switching Logic ---
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            panes.forEach(p => p.classList.add('hidden'));
            
            tab.classList.add('active');
            document.getElementById(tab.dataset.target).classList.remove('hidden');
        });
    });

    // --- Ego Network Logic ---
    egoForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const txId = egoInput.value.trim();
        if (!txId) return;

        resetGraphUI("Ego-Network Visualization (2-Hop)");
        resultsWidget.classList.add('hidden');
        
        try {
            const res = await fetch(`/wallet/${txId}/subgraph?hops=2`);
            if (!res.ok) throw new Error(res.status === 404 ? "Transaction not found." : "API Error");
            
            const data = await res.json();
            updateEgoWidget(data);
            renderGraph(data.nodes, data.edges, [txId]);
            
        } catch (error) {
            handleError(error);
        }
    });

    function updateEgoWidget(data) {
        const centerNode = data.nodes.find(n => n.txId === data.center) || data.nodes[0];
        const percentage = Math.round((centerNode.risk_score || 0) * 100);
        
        document.getElementById('score-text').textContent = `${percentage}%`;
        document.getElementById('score-circle').setAttribute('stroke-dasharray', `${percentage}, 100`);
        
        const label = centerNode.predicted_label || 'unknown';
        const pLabel = document.getElementById('prediction-label');
        pLabel.textContent = label;
        pLabel.className = `score-label ${label}`;
        
        let strokeColor = 'var(--color-unknown)';
        if (label === 'illicit') strokeColor = 'var(--color-illicit)';
        if (label === 'licit') strokeColor = 'var(--color-licit)';
        document.getElementById('score-circle').style.stroke = strokeColor;
        
        document.getElementById('stat-confidence').textContent = centerNode.confidence ? `${(centerNode.confidence * 100).toFixed(1)}%` : 'N/A';
        document.getElementById('stat-community').textContent = centerNode.communityId !== null ? centerNode.communityId : 'N/A';
        document.getElementById('stat-pagerank').textContent = centerNode.pageRank ? centerNode.pageRank.toFixed(4) : 'N/A';
        document.getElementById('stat-latency').textContent = `${data.latency_ms}ms`;
        
        const cacheBadge = document.getElementById('cache-badge');
        if (data.latency_ms < 15) {
            cacheBadge.textContent = 'HIT';
            cacheBadge.className = 'badge hit';
        } else {
            cacheBadge.textContent = 'MISS';
            cacheBadge.className = 'badge';
        }
        resultsWidget.classList.remove('hidden');
    }

    // --- Path Analysis Logic ---
    pathForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const src = srcInput.value.trim();
        const dst = dstInput.value.trim();
        if (!src || !dst) return;

        resetGraphUI(`Path Trace: ${src} ➔ ${dst}`);
        pathResults.classList.add('hidden');

        try {
            const res = await fetch(`/path?src=${src}&dst=${dst}`);
            if (!res.ok) throw new Error("API Error");
            
            const data = await res.json();
            
            document.getElementById('path-status-text').textContent = data.path_found ? "Path Found" : "No Path Exists";
            document.getElementById('path-status-text').style.color = data.path_found ? "var(--color-licit)" : "var(--color-illicit)";
            document.getElementById('path-length-text').textContent = data.path_found ? data.path_length : "N/A";
            document.getElementById('path-latency-text').textContent = `${data.latency_ms}ms`;
            
            pathResults.classList.remove('hidden');
            
            if (data.path_found) {
                // To render a graph of the path, we need to convert path_nodes into nodes and edges
                const nodes = data.path_nodes.map(id => ({ txId: id, predicted_label: 'unknown' })); // We don't get full node data from /path, so mock it visually
                const edges = [];
                for(let i=0; i<data.path_nodes.length-1; i++) {
                    edges.push({ source: data.path_nodes[i], target: data.path_nodes[i+1] });
                }
                renderGraph(nodes, edges, data.path_nodes, true);
            } else {
                handleError({message: "No path exists between these nodes within 10 hops."});
            }
            
        } catch (error) {
            handleError(error);
        }
    });

    // --- Cluster Logic ---
    btnLoadClusters.addEventListener('click', async () => {
        clusterResults.classList.add('hidden');
        clusterResults.innerHTML = '<div class="spinner" style="margin:20px auto;"></div>';
        clusterResults.classList.remove('hidden');
        
        try {
            const res = await fetch(`/cluster/top`);
            if (!res.ok) throw new Error("API Error fetching clusters");
            
            const data = await res.json();
            
            let html = `
                <table class="cluster-table">
                    <thead>
                        <tr>
                            <th>Community ID</th>
                            <th>Total Size</th>
                            <th>Illicit Count</th>
                            <th>Risk Ratio</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            
            data.clusters.forEach(c => {
                const ratio = (c.illicit_ratio * 100).toFixed(1);
                const isHighRisk = c.illicit_ratio > 0.1;
                html += `
                    <tr>
                        <td style="font-weight:bold; color:var(--accent-primary)">${c.communityId}</td>
                        <td>${c.total_nodes}</td>
                        <td style="color:${isHighRisk ? 'var(--color-illicit)' : 'inherit'}">${c.illicit_nodes}</td>
                        <td style="color:${isHighRisk ? 'var(--color-illicit)' : 'inherit'}">${ratio}%</td>
                    </tr>
                `;
            });
            html += `</tbody></table><div style="text-align:right; font-size:0.8rem; color:var(--text-secondary); margin-top:10px;">Query Latency: ${data.latency_ms}ms</div>`;
            
            clusterResults.innerHTML = html;
            
        } catch (error) {
            clusterResults.innerHTML = `<div style="color:var(--color-illicit); padding:1rem;">${error.message}</div>`;
        }
    });

    // --- Shared Graph Rendering ---
    function resetGraphUI(title) {
        graphTitle.textContent = title;
        placeholder.classList.add('hidden');
        svg.style('display', 'none');
        loading.classList.remove('hidden');
    }

    function handleError(error) {
        alert(error.message);
        placeholder.querySelector('p').textContent = error.message;
        placeholder.classList.remove('hidden');
        loading.classList.add('hidden');
    }

    function renderGraph(nodeData, edgeData, highlightedNodeIds = [], isPath = false) {
        loading.classList.add('hidden');
        svg.style('display', 'block');
        g.selectAll('*').remove();

        if (simulation) simulation.stop();

        const nodes = nodeData.map(d => ({ ...d, id: d.txId }));
        const links = edgeData.map(d => ({ source: d.source, target: d.target }));

        simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(isPath ? 100 : 60))
            .force('charge', d3.forceManyBody().strength(isPath ? -300 : -150))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collide', d3.forceCollide().radius(20));

        const link = g.append('g')
            .selectAll('line')
            .data(links)
            .enter().append('line')
            .attr('class', d => {
                if (isPath) return 'link path-highlight';
                return 'link';
            });

        const node = g.append('g')
            .selectAll('g')
            .data(nodes)
            .enter().append('g')
            .attr('class', d => highlightedNodeIds.includes(d.id) && isPath ? 'node path-highlight' : 'node')
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));

        node.append('circle')
            .attr('r', d => highlightedNodeIds.includes(d.id) ? (isPath ? 15 : 12) : 8)
            .attr('fill', d => {
                if (highlightedNodeIds.includes(d.id) && !isPath) return 'var(--color-target)';
                if (d.predicted_label === 'illicit') return 'var(--color-illicit)';
                if (d.predicted_label === 'licit') return 'var(--color-licit)';
                return 'var(--color-unknown)';
            })
            .on('mouseover', (event, d) => {
                tooltip.style('opacity', 1)
                    .html(`
                        <strong>TX: ${d.id}</strong><br>
                        Pred: ${d.predicted_label || 'unknown'}<br>
                        ${d.risk_score ? 'Risk: '+(d.risk_score*100).toFixed(1)+'%<br>' : ''}
                    `)
                    .style('left', (event.pageX + 15) + 'px')
                    .style('top', (event.pageY - 15) + 'px');
            })
            .on('mouseout', () => tooltip.style('opacity', 0))
            .on('click', (event, d) => {
                egoInput.value = d.id;
                document.querySelector('.tab-btn[data-target="tab-ego"]').click();
                egoForm.dispatchEvent(new Event('submit'));
            });

        if (isPath) {
            node.append('text')
                .attr('dx', 18)
                .attr('dy', 4)
                .text(d => d.id);
        }

        simulation.on('tick', () => {
            link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
            node.attr('transform', d => `translate(${d.x},${d.y})`);
        });

        function dragstarted(event) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }
        function dragged(event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }
        function dragended(event) {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }
    }
});
