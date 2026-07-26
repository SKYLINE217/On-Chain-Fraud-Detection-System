import express from 'express';
import jwt from 'jsonwebtoken';
import { z } from 'zod';
import { config } from '../config';

const router = express.Router();

const loginSchema = z.object({
  username: z.string(),
  password: z.string(),
});

router.post('/login', (req, res) => {
  try {
    const { username, password } = loginSchema.parse(req.body);

    if (username === config.ADMIN_USERNAME && password === config.ADMIN_PASSWORD) {
      const token = jwt.sign(
        { username, role: 'admin' },
        config.JWT_SECRET,
        { expiresIn: config.JWT_EXPIRES_IN }
      );
      res.json({ token });
    } else {
      res.status(401).json({ error: 'Invalid credentials' });
    }
  } catch (error) {
    if (error instanceof z.ZodError) {
      res.status(400).json({ error: 'Invalid request format' });
    } else {
      res.status(500).json({ error: 'Internal server error' });
    }
  }
});

export default router;
