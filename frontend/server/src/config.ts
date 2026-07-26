import dotenv from 'dotenv';
import { z } from 'zod';

// Load .env (in development, it's at project root, but this BFF also needs it)
// We rely on Docker Compose passing these env vars in production.
dotenv.config({ path: '../../.env' }); 

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
