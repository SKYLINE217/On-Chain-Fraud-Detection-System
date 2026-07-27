import { useEffect, useRef } from 'react';

export function TopologyGrid() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    let animationFrameId: number;
    let nodes: Array<{x: number, y: number, vx: number, vy: number, r: number, color: string}> = [];
    
    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = 400; // Matches hero height
    };
    
    window.addEventListener('resize', resize);
    resize();

    const colors = [
      ...Array(80).fill('#22C55E'),
      ...Array(15).fill('#F59E0B'),
      ...Array(5).fill('#EF4444')
    ];

    for (let i = 0; i < 40; i++) {
      nodes.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * (prefersReducedMotion ? 0 : 0.5),
        vy: (Math.random() - 0.5) * (prefersReducedMotion ? 0 : 0.5),
        r: Math.random() * 2 + 2,
        color: colors[Math.floor(Math.random() * colors.length)]
      });
    }

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      for (let i = 0; i < nodes.length; i++) {
        let n = nodes[i];
        n.x += n.vx;
        n.y += n.vy;
        
        if (n.x < 0 || n.x > canvas.width) n.vx *= -1;
        if (n.y < 0 || n.y > canvas.height) n.vy *= -1;

        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fillStyle = n.color;
        ctx.fill();
        
        for (let j = i + 1; j < nodes.length; j++) {
          let n2 = nodes[j];
          const dist = Math.hypot(n.x - n2.x, n.y - n2.y);
          if (dist < 80) {
            ctx.beginPath();
            ctx.moveTo(n.x, n.y);
            ctx.lineTo(n2.x, n2.y);
            ctx.strokeStyle = `rgba(42, 51, 80, ${1 - dist / 80})`;
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
      }
      
      if (!prefersReducedMotion) {
        animationFrameId = requestAnimationFrame(draw);
      }
    };

    draw();

    return () => {
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-[400px] pointer-events-none opacity-50 z-0" />;
}
