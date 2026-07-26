import { createProxyMiddleware } from 'http-proxy-middleware';
import { config } from './config';
import { logger } from './middleware/errorHandler';

// Proxy configuration to forward requests to the FastAPI backend
export const fastApiProxy = createProxyMiddleware({
  target: config.FASTAPI_URL,
  changeOrigin: true,
  pathRewrite: {
    '^/api': '', // remove /api prefix when forwarding to FastAPI
  },
  on: {
    proxyReq: (proxyReq, req, res) => {
      // Inject the FastAPI API key on all proxied requests
      proxyReq.setHeader('X-API-Key', config.FASTAPI_API_KEY);
    },
    error: (err, req, res) => {
      logger.error(`Proxy Error: ${err.message}`);
      res.writeHead(503, {
        'Content-Type': 'application/json',
      });
      res.end(JSON.stringify({ error: 'Backend service unavailable' }));
    }
  }
});
