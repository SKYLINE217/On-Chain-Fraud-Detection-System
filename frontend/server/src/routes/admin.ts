import express from 'express';
import jwt from 'jsonwebtoken';
import { z } from 'zod';
import { config } from '../config';
import { timingSafeEqual } from 'crypto';

const router = express.Router();

// AC-02: Account Lockout Mechanism
const loginAttempts = new Map<string, { count: number, lockUntil: number }>();
const MAX_ATTEMPTS = 5;
const LOCKOUT_MS = 15 * 60 * 1000; // 15 minutes

const loginSchema = z.object({
  username: z.string(),
  password: z.string(),
});

router.post('/login', (req, res) => {
  try {
    const ip = req.ip || req.socket.remoteAddress || 'unknown';
    
    // AC-02: Check if locked out
    const attempts = loginAttempts.get(ip);
    if (attempts && attempts.lockUntil > Date.now()) {
      return res.status(429).json({ error: 'Account locked. Try again later.' });
    }

    const { username, password } = loginSchema.parse(req.body);

    // AC-08: Use timingSafeEqual to prevent timing attacks
    // We pad or hash strings if they are different lengths, but simpler is comparing hashes or padded buffers
    // To safely compare strings of unknown length, we can hash them first or ensure they are same length.
    // crypto.timingSafeEqual requires same length buffers.
    const expectedUsername = Buffer.from(config.ADMIN_USERNAME);
    const providedUsername = Buffer.from(username);
    const expectedPassword = Buffer.from(config.ADMIN_PASSWORD);
    const providedPassword = Buffer.from(password);

    const usernameMatch = providedUsername.length === expectedUsername.length && 
                          timingSafeEqual(providedUsername, expectedUsername);
    const passwordMatch = providedPassword.length === expectedPassword.length && 
                          timingSafeEqual(providedPassword, expectedPassword);

    if (usernameMatch && passwordMatch) {
      // Clear attempts on success
      loginAttempts.delete(ip);

      const token = jwt.sign(
        { username, role: 'admin' },
        config.JWT_SECRET,
        { expiresIn: config.JWT_EXPIRES_IN as any }
      );
      
      // AC-04: Store JWT in httpOnly cookie
      res.cookie('authToken', token, {
        httpOnly: true,
        secure: config.NODE_ENV === 'production',
        sameSite: 'strict',
        // MaxAge expects ms, JWT_EXPIRES_IN is a string ('8h'), we assume 8h in ms for the cookie
        maxAge: 8 * 60 * 60 * 1000, 
      });
      return res.json({ success: true });
    } else {
      // Record failed attempt
      const current = loginAttempts.get(ip) || { count: 0, lockUntil: 0 };
      current.count += 1;
      if (current.count >= MAX_ATTEMPTS) {
        current.lockUntil = Date.now() + LOCKOUT_MS;
      }
      loginAttempts.set(ip, current);

      return res.status(401).json({ error: 'Invalid credentials' });
    }
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({ error: 'Invalid request format' });
    } else {
      return res.status(500).json({ error: 'Internal server error' });
    }
  }
});

export default router;
