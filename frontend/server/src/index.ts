import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import path from 'path';
import { config } from './config';
import { fastApiProxy } from './proxy';
import { standardLimiter, adminLimiter } from './middleware/rateLimit';
import { errorHandler, logger } from './middleware/errorHandler';
import adminAuthRouter from './routes/admin';
import { verifyToken } from './middleware/auth';

const app = express();

// Security Middleware
app.use(helmet({
  hsts: { maxAge: 31536000, includeSubDomains: true, preload: true }, // BFF-03: HSTS
  frameguard: { action: 'deny' }, // BFF-03: X-Frame-Options
  referrerPolicy: { policy: 'strict-origin-when-cross-origin' }, // BFF-03
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'"], // FE-01: Remove 'unsafe-inline'
      styleSrc: ["'self'", "'unsafe-inline'"], // Tailwind requires this, but limited to self
      imgSrc: ["'self'", "data:", "blob:"],
      connectSrc: ["'self'", ...config.ALLOWED_ORIGINS.split(',')],
      objectSrc: ["'none'"],
      upgradeInsecureRequests: [],
    }
  }
}));

const allowedOrigins = config.ALLOWED_ORIGINS.split(',');
app.use(cors({
  origin: (origin, callback) => {
    // BFF-02: Fix CORS `!origin` bypass
    if (config.NODE_ENV === 'development' && !origin) {
      return callback(null, true); // Allow curl/postman in dev
    }
    if (origin && allowedOrigins.indexOf(origin) !== -1) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true,
}));

// Body parsing (ONLY for non-proxied routes)
// BFF-07: We intentionally DO NOT use express.json() globally.
// The /api proxy (http-proxy-middleware) requires the raw unparsed request stream
// to correctly forward POST bodies (e.g., /api/score) to FastAPI.
// Applying express.json() globally would consume the stream and break the proxy.
app.use('/api/admin/auth', express.json());

// Routes
// 1. Health check (BFF level)
app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'bff', environment: config.NODE_ENV });
});

// 2. Admin Auth
app.use('/api/admin/auth', adminLimiter, adminAuthRouter);

// 3. Proxy to FastAPI (Wallet, Cluster, Explain, Score)
// Any route starting with /api (except auth) goes to FastAPI
// AC-01: Apply verifyToken to require JWT for all proxy routes
// except a specific allow-list (like health)
const PUBLIC_PATHS = ['/api/health'];
app.use('/api', standardLimiter, (req, res, next) => {
  if (PUBLIC_PATHS.includes(req.path)) {
    return next();
  }
  return verifyToken(req, res, next);
}, fastApiProxy);

// 4. Serve React Client (Production)
if (config.NODE_ENV === 'production') {
  app.use(express.static(path.join(__dirname, '../../client/dist')));
  app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, '../../client/dist/index.html'));
  });
} else {
  app.get('/', (req, res) => {
    res.send('BFF API running in development mode. Start React client separately.');
  });
}

// Error Handling
app.use(errorHandler);

const PORT = config.PORT;
app.listen(PORT, () => {
  logger.info(`BFF server running on port ${PORT} in ${config.NODE_ENV} mode`);
});
