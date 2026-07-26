import dotenv from 'dotenv';
import { z } from 'zod';
import path from 'path';
import fs from 'fs';

// BFF-04: Robust dotenv path resolution
const envPaths = [
  path.resolve(__dirname, '../../.env'),      // Docker build path
  path.resolve(process.cwd(), '.env'),         // Local dev
  path.resolve(process.cwd(), '../.env'),      // Alt local path
];

for (const envPath of envPaths) {
  if (fs.existsSync(envPath)) {
    dotenv.config({ path: envPath });
    break;
  }
}

const envSchema = z.object({
  NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),
  PORT: z.coerce.number().default(3000),
  FASTAPI_URL: z.string().url().default('http://localhost:8000'),
  FASTAPI_API_KEY: z.string().min(1),
  JWT_SECRET: z.string().min(16),
  JWT_EXPIRES_IN: z.string().default('8h'),
  ADMIN_USERNAME: z.string().min(1),
  ADMIN_PASSWORD: z.string().min(1),
  RATE_LIMIT_WINDOW_MS: z.coerce.number().default(60000),
  RATE_LIMIT_MAX: z.coerce.number().default(100),
  ADMIN_RATE_LIMIT_MAX: z.coerce.number().default(20),
  ALLOWED_ORIGINS: z.string().default('http://localhost:3000,http://localhost:5173')
});

const parsed = envSchema.safeParse(process.env);

if (!parsed.success) {
  console.error('❌ Invalid environment variables:', parsed.error.format());
  process.exit(1);
}

export const config = parsed.data;

// AC-02: Enforce changing default admin password in production
if (config.ADMIN_PASSWORD === 'admin' && config.NODE_ENV === 'production') {
  console.error('❌ FATAL: Default admin password must be changed in production!');
  process.exit(1);
}
