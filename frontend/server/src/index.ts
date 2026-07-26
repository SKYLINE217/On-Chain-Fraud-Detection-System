import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import path from 'path';
import { config } from './config';
import { fastApiProxy } from './proxy';
import { standardLimiter, adminLimiter } from './middleware/rateLimit';
import { errorHandler, logger } from './middleware/errorHandler';
import adminAuthRouter from './routes/admin';

const app = express();

// Security Middleware
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'", "'unsafe-inline'"], // Needed for some React/Vite dev setups, adjust for strict prod
      styleSrc: ["'self'", "'unsafe-inline'"],
      imgSrc: ["'self'", "data:", "blob:"],
      connectSrc: ["'self'", ...config.ALLOWED_ORIGINS.split(',')],
    }
  }
}));

const allowedOrigins = config.ALLOWED_ORIGINS.split(',');
app.use(cors({
  origin: (origin, callback) => {
    if (!origin || allowedOrigins.indexOf(origin) !== -1) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true,
}));

// Body parsing (ONLY for non-proxied routes)
// We don't use express.json() globally because http-proxy-middleware needs the raw stream
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
app.use('/api', standardLimiter, fastApiProxy);

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
