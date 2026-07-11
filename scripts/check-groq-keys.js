#!/usr/bin/env node
require('dotenv').config();

const keys = (process.env.GROQ_API_KEY || '')
  .split(',')
  .map((k) => k.trim())
  .filter(Boolean);

async function probeKey(apiKey, index) {
  const prefix = apiKey.slice(0, 8) + '...' + apiKey.slice(-4);
  const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: 'qwen/qwen3.6-27b',
      messages: [{ role: 'user', content: 'Reply with exactly: OK' }],
      max_tokens: 8,
      reasoning_effort: 'none'
    })
  });

  const text = await res.text();
  let org = null;
  const orgMatch = text.match(/organization `([^`]+)`/);
  if (orgMatch) org = orgMatch[1];

  const headerOrg = res.headers.get('x-groq-organization') || res.headers.get('groq-organization');
  if (headerOrg) org = headerOrg;

  if (res.ok) {
    return { index, prefix, status: 'OK', org: org || '(not in response — keys may share quota silently)' };
  }

  const short = res.status === 429 ? 'RATE LIMITED (Qwen TPD)' : `HTTP ${res.status}`;
  return { index, prefix, status: short, org: org || 'unknown', detail: text.slice(0, 120) };
}

async function main() {
  console.log(`Checking ${keys.length} Groq API keys (Qwen 3.6, reasoning off)...\n`);
  const orgs = new Map();

  for (let i = 0; i < keys.length; i++) {
    try {
      const result = await probeKey(keys[i], i);
      const orgLabel = result.org ? ` | org=${result.org}` : '';
      console.log(`Key ${result.index} (${result.prefix}): ${result.status}${orgLabel}`);
      if (result.org && result.org !== 'unknown') {
        const list = orgs.get(result.org) || [];
        list.push(i);
        orgs.set(result.org, list);
      }
    } catch (e) {
      console.log(`Key ${i}: ERROR ${e.message}`);
    }
  }

  console.log('\n--- Org summary ---');
  if (orgs.size === 0) {
    console.log('No org IDs found in responses. If all keys work, Groq does not expose org in success responses.');
    console.log('Rate-limit errors (429) on Qwen will show org_... in the message — that is the reliable check.');
  } else {
    for (const [org, keyIndexes] of orgs) {
      console.log(`${org} → keys [${keyIndexes.join(', ')}]`);
    }
    console.log(`\n${orgs.size} unique org(s) across ${keys.length} keys`);
    if (orgs.size < keys.length) {
      console.log('Note: keys without 429 errors may belong to different orgs — Groq only reveals org on rate-limit messages.');
    }
  }
}

main();
