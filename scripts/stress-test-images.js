#!/usr/bin/env node
require('dotenv').config();

const fs = require('fs');
const path = require('path');

const RUNS = parseInt(process.argv[2] || '6', 10);
const ASSETS = path.join(__dirname, '../.cursor-test-assets');
const IMAGES = [
  { name: 'shape-puzzle', file: 'puzzle.png' },
  { name: 'python-mcq', file: 'python-mcq.png' }
].filter((img) => fs.existsSync(path.join(ASSETS, img.file)));

function summarize(results) {
  const ok = results.filter((r) => r.ok);
  const fail = results.filter((r) => !r.ok);
  const times = ok.map((r) => r.ms);
  const avg = times.length ? Math.round(times.reduce((a, b) => a + b, 0) / times.length) : 0;
  const min = times.length ? Math.min(...times) : 0;
  const max = times.length ? Math.max(...times) : 0;
  const rateLimits = results.filter((r) => /rate limit|429/i.test(r.error || ''));
  return { ok: ok.length, fail: fail.length, avg, min, max, rateLimits: rateLimits.length, failures: fail };
}

async function runOnce(llmService, sessionManager, imagePath, label, runNum) {
  const imageBuffer = fs.readFileSync(imagePath);
  const start = Date.now();
  try {
    sessionManager.setResponseMode('complex');
    const result = await llmService.processImageWithSkill(imageBuffer, 'image/png', 'general', [], null);
    const ms = Date.now() - start;
    const preview = (result.response || '').replace(/\s+/g, ' ').slice(0, 80);
    const valid = result.response && result.response.trim().length > 5 && !/could not solve|vision model error/i.test(result.response);
    console.log(`  [${label} #${runNum}] ${valid ? 'OK' : 'WEAK'} ${ms}ms — ${preview}`);
    return { ok: valid, ms, preview, response: result.response, metadata: result.metadata };
  } catch (err) {
    const ms = Date.now() - start;
    console.log(`  [${label} #${runNum}] FAIL ${ms}ms — ${err.message}`);
    return { ok: false, ms, error: err.message };
  }
}

async function main() {
  if (!process.env.GROQ_API_KEY) {
    console.error('GROQ_API_KEY not set');
    process.exit(1);
  }
  if (IMAGES.length === 0) {
    console.error('No test images in', ASSETS);
    process.exit(1);
  }

  const llmService = require('../src/services/llm.service');
  const sessionManager = require('../src/managers/session.manager');
  const keyCount = llmService.clients.length;

  console.log(`\n=== STRESS TEST: ${RUNS} runs × ${IMAGES.length} images (${keyCount} API keys) ===\n`);

  const allResults = {};
  const totalStart = Date.now();

  for (const img of IMAGES) {
    const imagePath = path.join(ASSETS, img.file);
    allResults[img.name] = [];
    console.log(`\n--- ${img.name} (${img.file}) ---`);
    for (let i = 1; i <= RUNS; i++) {
      const result = await runOnce(llmService, sessionManager, imagePath, img.name, i);
      allResults[img.name].push(result);
      // tiny gap so round-robin advances between runs
      await new Promise((r) => setTimeout(r, 300));
    }
    const stats = summarize(allResults[img.name]);
    console.log(`  → ${stats.ok}/${RUNS} ok | avg ${stats.avg}ms (${stats.min}-${stats.max}) | rate-limits: ${stats.rateLimits}`);
    if (stats.failures.length) {
      stats.failures.forEach((f) => console.log(`     ✗ ${f.error || f.preview}`));
    }
  }

  const totalMs = Date.now() - totalStart;
  const flat = Object.values(allResults).flat();
  const total = summarize(flat);
  console.log(`\n=== TOTAL: ${total.ok}/${flat.length} ok | ${total.rateLimits} rate-limits | ${Math.round(totalMs / 1000)}s wall time ===\n`);
  process.exit(total.fail > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
