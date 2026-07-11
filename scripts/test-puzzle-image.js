#!/usr/bin/env node
require('dotenv').config();

const fs = require('fs');
const path = require('path');

const IMAGE = process.argv[2] || path.join(
  __dirname,
  '../.cursor-test-assets/puzzle.png'
);

async function main() {
  if (!process.env.GROQ_API_KEY) {
    console.error('GROQ_API_KEY not set — load .env or pass keys in env');
    process.exit(1);
  }

  if (!fs.existsSync(IMAGE)) {
    console.error('Image not found:', IMAGE);
    process.exit(1);
  }

  const llmService = require('../src/services/llm.service');
  const sessionManager = require('../src/managers/session.manager');

  sessionManager.setResponseMode('complex');

  const imageBuffer = fs.readFileSync(IMAGE);
  console.log('Testing image:', IMAGE, `(${imageBuffer.length} bytes)`);

  const start = Date.now();
  const result = await llmService.processImageWithSkill(
    imageBuffer,
    'image/png',
    'general',
    [],
    null
  );

  console.log('\n--- RESULT ---');
  console.log('Time:', Date.now() - start, 'ms');
  console.log('Metadata:', JSON.stringify(result.metadata, null, 2));
  console.log('Response:\n', result.response);
}

main().catch((err) => {
  console.error('TEST FAILED:', err.message);
  process.exit(1);
});
