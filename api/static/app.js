document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('search-form');
    const input = document.getElementById('tx-id');
    const resultsWidget = document.getElementById('results-widget');
    const placeholder = document.getElementById('graph-placeholder');
    const loading = document.getElementById('graph-loading');
    const graphContainer = document.getElementById('graph-container');
    
    // Stats elements
    const scoreCircle = document.getElementById('score-circle');
    const scoreText = document.getElementById('score-text');
    const predLabel = document.getElementById('prediction-label');
    const cacheBadge = document.getElementById('cache-badge');
    const statConf = document.getElementById('stat-confidence');
    const statComm = document.getElementById('stat-community');
    const statPage = document.getElementById('stat-pagerank');
    const statLatency = document.getElementById('stat-latency');

    // D3 Setup
    const width = graphContainer.clientWidth;
    const height = graphContainer.clientHeight;
    
    const svg = d3.select('#graph-container')
        .append('svg')
        .attr('width', '100%')
        .attr('height', '100%')
        .style('display', 'none');
        
    // Add zoom capabilities
    const g = svg.append('g');
    svg.call(d3.zoom().on('zoom', (event) => {
        g.attr('transform', event.transform);
    }));

    // Tooltip
    const tooltip = d3.select('body').append('div')
        .attr('class', 'd3-tooltip');

    let simulation = null;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const txId = input.value.trim();
        if (!txId) return;

        // UI Reset
        resultsWidget.classList.add('hidden');
        placeholder.classList.add('hidden');
        svg.style('display', 'none');
        loading.classList.remove('hidden');
        
        try {
            // Fetch graph data from our FastAPI endpoint
            const res = await fetch(`/wallet/${txId}/subgraph?hops=2`);
            if (!res.ok) {
                if(res.status === 404) {
                    throw new Error("Transaction not found or has no connections.");
                }
                throw new Error(`API Error: ${res.statusText}`);
            }
            
            const data = await res.json();
            
            // Render Results Widget
            updateWidget(data);
            
            // Render Graph
            renderGraph(data, txId);
            
        } catch (error) {
            alert(error.message);
            placeholder.classList.remove('hidden');
            loading.classList.add('hidden');
        }
    });

    function updateWidget(data) {
        // Find the center node
        const centerNode = data.nodes.find(n => n.txId === data.center) || data.nodes[0];
        
        // Update Risk Score (Circular progress)
        const score = centerNode.risk_score || 0;
        const percentage = Math.round(score * 100);
        
        scoreText.textContent = `${percentage}%`;
        scoreCircle.setAttribute('stroke-dasharray', `${percentage}, 100`);
        
        // Colors based on predicted label
        const label = centerNode.predicted_label || 'unknown';
        predLabel.textContent = label;
        predLabel.className = `score-label ${label}`;
        
        let strokeColor = 'var(--color-unknown)';
        if (label === 'illicit') strokeColor = 'var(--color-illicit)';
        if (label === 'licit') strokeColor = 'var(--color-licit)';
        scoreCircle.style.stroke = strokeColor;
        
        // Stats
        statConf.textContent = centerNode.confidence ? `${(centerNode.confidence * 100).toFixed(1)}%` : 'N/A';
        statComm.textContent = centerNode.communityId !== null ? centerNode.communityId : 'N/A';
        statPage.textContent = centerNode.pageRank ? centerNode.pageRank.toFixed(4) : 'N/A';
        
        // Latency & Cache
        statLatency.textContent = `${data.latency_ms}ms`;
        // Since /subgraph endpoint doesn't return 'cached' boolean in the current API,
        // we'll simulate a cache badge based on latency (<10ms is likely cached/redis)
        if (data.latency_ms < 15) {
            cacheBadge.textContent = 'HIT';
            cacheBadge.className = 'badge hit';
        } else {
            cacheBadge.textContent = 'MISS';
            cacheBadge.className = 'badge';
        }

        resultsWidget.classList.remove('hidden');
    }

    function renderGraph(data, targetTxId) {
        loading.classList.add('hidden');
        svg.style('display', 'block');
        g.selectAll('*').remove();

        if (simulation) simulation.stop();

        // Create node/link maps for D3
        const nodes = data.nodes.map(d => ({ ...d, id: d.txId }));
        const links = data.edges.map(d => ({ source: d.source, target: d.target }));

        simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(60))
            .force('charge', d3.forceManyBody().strength(-150))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collide', d3.forceCollide().radius(20));

        // Draw links
        const link = g.append('g')
            .selectAll('line')
            .data(links)
            .enter().append('line')
            .attr('class', 'link');

        // Draw nodes
        const node = g.append('g')
            .selectAll('g')
            .data(nodes)
            .enter().append('g')
            .attr('class', 'node')
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));

        node.append('circle')
            .attr('r', d => d.id === targetTxId ? 12 : 8)
            .attr('fill', d => {
                if (d.id === targetTxId) return 'var(--color-target)';
                if (d.predicted_label === 'illicit') return 'var(--color-illicit)';
                if (d.predicted_label === 'licit') return 'var(--color-licit)';
                return 'var(--color-unknown)';
            })
            .on('mouseover', (event, d) => {
                tooltip.style('opacity', 1)
                    .html(`
                        <strong>TX: ${d.id}</strong><br>
                        Class: ${d.txClass || 'Unknown'}<br>
                        Pred: ${d.predicted_label || 'N/A'}<br>
                        Risk: ${d.risk_score ? (d.risk_score*100).toFixed(1)+'%' : 'N/A'}<br>
                        Community: ${d.communityId || 'N/A'}
                    `)
                    .style('left', (event.pageX + 15) + 'px')
                    .style('top', (event.pageY - 15) + 'px');
            })
            .on('mouseout', () => {
                tooltip.style('opacity', 0);
            })
            .on('click', (event, d) => {
                // Clicking a node auto-searches it
                input.value = d.id;
                form.dispatchEvent(new Event('submit'));
            });

        simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            node
                .attr('transform', d => `translate(${d.x},${d.y})`);
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
