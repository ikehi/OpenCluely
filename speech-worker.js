'use strict';

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const WebSocket = require('ws');

// ---------------------------------------------------------------------------
// Logging helper – sends structured logs to the parent process
// ---------------------------------------------------------------------------
function log(level, message, data) {
  try {
    process.send({ type: 'log', level, message, data: data || {} });
  } catch (_) {
    process.stderr.write(`[speech-worker] ${level}: ${message} ${JSON.stringify(data || {})}\n`);
  }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let isRecording = false;
let sessionStartTime = null;
let recordingProcess = null; // sox child process
let available = false;

// Deepgram streaming state
let deepgramApiKey = null;
let dgSocket = null;

// Groq fallback state
let groqClients = [];
let currentClientIndex = 0;
let useDeepgram = false; // true when DEEPGRAM_API_KEY is present

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
function initialize(config) {
  try {
    // Check for Deepgram key first (preferred)
    if (config.deepgramKey && config.deepgramKey.trim().length > 0) {
      deepgramApiKey = config.deepgramKey.trim();
      useDeepgram = true;
      available = true;
      log('info', 'Deepgram streaming mode enabled', { keyLength: deepgramApiKey.length });
      process.send({ type: 'init-result', available: true, mode: 'deepgram' });
      return;
    }

    // Fallback to Groq Whisper
    if (!config.groqKeys || config.groqKeys.length === 0) {
      available = false;
      process.send({ type: 'init-result', available: false, reason: 'Missing both DEEPGRAM_API_KEY and GROQ_API_KEY' });
      return;
    }

    const Groq = require('groq-sdk');
    groqClients = config.groqKeys.map(key => new Groq({ apiKey: key }));
    currentClientIndex = groqClients.length > 1 ? 1 : 0;
    useDeepgram = false;
    available = true;
    log('info', 'Groq Whisper fallback mode enabled', { keyCount: groqClients.length, startingIndex: currentClientIndex });
    process.send({ type: 'init-result', available: true, mode: 'groq' });
  } catch (error) {
    available = false;
    log('error', 'Failed to initialize speech worker', { error: error.message });
    process.send({ type: 'init-result', available: false, reason: error.message });
  }
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------
function cleanup() {
  if (recordingProcess) {
    try { recordingProcess.kill('SIGKILL'); } catch (_) { }
    recordingProcess = null;
  }
  if (dgSocket) {
    try {
      if (dgSocket.readyState === WebSocket.OPEN) {
        // Send close_stream message to gracefully close
        dgSocket.send(JSON.stringify({ type: 'CloseStream' }));
      }
      dgSocket.close();
    } catch (_) { }
    dgSocket = null;
  }
}

// ===========================================================================
//  DEEPGRAM STREAMING MODE (raw WebSocket — no SDK wrapper)
// ===========================================================================

function startDeepgramStreaming() {
  // Build the Deepgram streaming URL with query parameters
  const params = new URLSearchParams({
    model: 'nova-3',
    language: 'en',
    encoding: 'linear16',
    sample_rate: '16000',
    channels: '1',
    punctuate: 'true',
    interim_results: 'true',
    utterance_end_ms: '1000',
    vad_events: 'true',
    smart_format: 'true',
  });

  const url = `wss://api.deepgram.com/v1/listen?${params.toString()}`;

  log('info', 'Opening Deepgram WebSocket connection...', { url: url.replace(deepgramApiKey, '***') });

  dgSocket = new WebSocket(url, {
    headers: {
      Authorization: `Token ${deepgramApiKey}`,
    },
  });

  dgSocket.on('open', () => {
    log('info', 'Deepgram WebSocket connection opened');
    // Now start sox to capture microphone audio as raw PCM to stdout
    spawnSoxStreaming();
  });

  dgSocket.on('message', (raw) => {
    try {
      const data = JSON.parse(raw.toString());

      if (data.type === 'Results') {
        const transcript = data.channel?.alternatives?.[0]?.transcript;
        if (!transcript || transcript.trim().length === 0) return;

        const isFinal = data.is_final;

        if (isFinal) {
          const dur = Date.now() - sessionStartTime;
          log('info', 'Final transcription', { text: transcript.trim(), sessionDuration: `${dur}ms` });
          process.send({ type: 'transcription', text: transcript.trim() });
        } else {
          // Interim result — show real-time words appearing
          process.send({ type: 'interim-transcription', text: transcript.trim() });
        }
      } else if (data.type === 'UtteranceEnd') {
        log('info', 'Deepgram utterance end detected — speaker finished');
        process.send({ type: 'utterance-end' });
      } else if (data.type === 'SpeechStarted') {
        log('debug', 'Deepgram speech started');
      } else if (data.type === 'Metadata') {
        log('debug', 'Deepgram metadata received', { request_id: data.request_id });
      } else if (data.type === 'Error') {
        log('error', 'Deepgram returned an error', { message: data.message, description: data.description });
      }
    } catch (err) {
      log('error', 'Error parsing Deepgram message', { error: err.message });
    }
  });

  dgSocket.on('error', (error) => {
    log('error', 'Deepgram WebSocket error', { error: error?.message || String(error) });
  });

  dgSocket.on('close', (code, reason) => {
    log('info', 'Deepgram WebSocket connection closed', { code, reason: reason?.toString() });
    dgSocket = null;

    // If still recording, attempt reconnection
    if (isRecording) {
      log('info', 'Reconnecting Deepgram WebSocket in 1s...');
      if (recordingProcess) {
        try { recordingProcess.kill('SIGKILL'); } catch (_) { }
        recordingProcess = null;
      }
      setTimeout(() => {
        if (isRecording) startDeepgramStreaming();
      }, 1000);
    }
  });
}

function spawnSoxStreaming() {
  const isWindows = process.platform === 'win32';
  const cmd = 'sox';
  let args;

  if (isWindows) {
    // Capture from Windows default audio device, output raw PCM to stdout
    args = [
      '-t', 'waveaudio', 'default', '-q',
      '-b', '16', '-e', 'signed', '-c', '1', '-r', '16000',
      '-t', 'raw', '-'
    ];
  } else {
    // macOS / Linux
    args = [
      '-d', '-q',
      '-b', '16', '-e', 'signed', '-c', '1', '-r', '16000',
      '-t', 'raw', '-'
    ];
  }

  recordingProcess = spawn(cmd, args);

  recordingProcess.stdout.on('data', (chunk) => {
    // Pipe raw PCM audio directly into the Deepgram WebSocket
    if (dgSocket && dgSocket.readyState === WebSocket.OPEN) {
      try {
        dgSocket.send(chunk);
      } catch (err) {
        log('debug', 'Failed to send audio chunk to Deepgram', { error: err.message });
      }
    }
  });

  recordingProcess.stderr.on('data', (data) => {
    const msg = data.toString().trim();
    if (msg.length > 0) {
      log('debug', 'Sox stderr', { message: msg });
    }
  });

  recordingProcess.on('error', (error) => {
    log('error', 'Failed to spawn sox for streaming', { error: error.message });
    if (isRecording) {
      process.send({ type: 'error', error: `Microphone capture failed (sox error): ${error.message}` });
      stopRecording();
    }
  });

  recordingProcess.on('close', (code) => {
    recordingProcess = null;
    log('debug', 'Sox streaming process closed', { code });
  });

  log('info', 'Sox streaming process spawned (raw PCM → Deepgram WebSocket)');
}

// ===========================================================================
//  GROQ WHISPER FALLBACK MODE (existing batch logic)
// ===========================================================================

async function runGroqRecordingLoop() {
  if (!isRecording) return;

  const tempWavPath = path.join(__dirname, 'temp_audio.wav');
  const isWindows = process.platform === 'win32';
  const cmd = 'sox';
  let args = [];

  const formatArgs = ['-b', '16', '-e', 'signed', '-c', '1', '-r', '16000', tempWavPath, 'silence', '1', '0.1', '1%', '1', '0.99', '1%'];

  if (isWindows) {
    args = ['-t', 'waveaudio', 'default', '-q', ...formatArgs];
  } else {
    args = ['-d', '-q', ...formatArgs];
  }

  recordingProcess = spawn(cmd, args);

  recordingProcess.on('error', (error) => {
    log('error', 'Failed to spawn sox', { error: error.message });
    if (isRecording) {
      process.send({ type: 'error', error: `Microphone capture failed (sox error): ${error.message}` });
      stopRecording();
    }
  });

  recordingProcess.on('close', async (code) => {
    recordingProcess = null;

    if (!isRecording) return;

    if (code !== 0 && code !== null) {
      log('warn', `sox exited with code ${code}`);
    }

    try {
      if (fs.existsSync(tempWavPath)) {
        const stats = fs.statSync(tempWavPath);
        if (stats.size > 2000) {
          log('debug', 'Uploading audio to Groq Whisper...', { size: stats.size, keyCount: groqClients.length });

          process.send({ type: 'interim-transcription', text: 'Transcribing...' });

          let transcription = null;
          let lastTranscriptionError = null;

          for (let attempt = 0; attempt < groqClients.length; attempt++) {
            const keyIndex = (currentClientIndex + attempt) % groqClients.length;
            try {
              transcription = await groqClients[keyIndex].audio.transcriptions.create({
                file: fs.createReadStream(tempWavPath),
                model: 'whisper-large-v3-turbo',
                response_format: 'text',
                language: 'en'
              });
              currentClientIndex = keyIndex;
              break;
            } catch (err) {
              lastTranscriptionError = err;
              const retryable = err.status === 429 ||
                err.status === 401 ||
                err.status === 403 ||
                (err.status && err.status >= 500) ||
                (err.message && /fetch failed|network|timeout|rate limit/i.test(err.message));

              log('warn', `Whisper failed on API key index ${keyIndex}`, {
                error: err.message,
                status: err.status,
                attempt: attempt + 1,
                totalKeys: groqClients.length,
                willRetry: retryable && attempt < groqClients.length - 1
              });

              if (!retryable || attempt === groqClients.length - 1) {
                break;
              }
            }
          }

          if (transcription && transcription.trim().length > 0) {
            const dur = Date.now() - sessionStartTime;
            log('info', 'Final transcription', { text: transcription.trim(), sessionDuration: `${dur}ms` });
            process.send({ type: 'transcription', text: transcription.trim() });
          } else if (lastTranscriptionError) {
            throw lastTranscriptionError;
          }
        }
      }
    } catch (err) {
      log('error', 'Groq transcription failed across API keys', {
        error: err.message,
        status: err.status,
        keyCount: groqClients.length
      });
    }

    // Loop!
    if (isRecording) {
      setTimeout(() => runGroqRecordingLoop(), 10);
    }
  });
}

// ===========================================================================
//  Public recording controls
// ===========================================================================

function startRecording() {
  if (!available) {
    process.send({ type: 'error', error: 'Speech service not initialized' });
    return;
  }
  if (isRecording) {
    log('warn', 'Recording already in progress');
    return;
  }
  isRecording = true;
  sessionStartTime = Date.now();
  process.send({ type: 'recording-started' });
  process.send({ type: 'session-started', sessionId: (useDeepgram ? 'dg-' : 'groq-') + Date.now() });

  cleanup();

  if (useDeepgram) {
    try {
      startDeepgramStreaming();
    } catch (err) {
      log('error', 'Failed to start Deepgram streaming', { error: err.message });
      process.send({ type: 'error', error: `Deepgram streaming failed: ${err.message}` });
      stopRecording();
    }
  } else {
    runGroqRecordingLoop();
  }
}

function stopRecording() {
  if (!isRecording) return;
  isRecording = false;
  cleanup();

  const dur = sessionStartTime ? Date.now() - sessionStartTime : 0;
  log('info', 'Stopping speech recognition', { sessionDuration: `${dur}ms`, mode: useDeepgram ? 'deepgram' : 'groq' });

  process.send({ type: 'recording-stopped' });
  process.send({ type: 'session-stopped', sessionId: (useDeepgram ? 'dg-' : 'groq-') + Date.now() });
}

function getStatus() {
  return {
    isRecording,
    isInitialized: available,
    available,
    mode: useDeepgram ? 'deepgram' : 'groq',
    sessionDuration: sessionStartTime ? Date.now() - sessionStartTime : 0,
  };
}

function testConnection() {
  if (!available) {
    process.send({ type: 'test-result', success: false, message: 'Speech service not initialized' });
    return;
  }
  process.send({ type: 'test-result', success: true, message: `Connection test successful (${useDeepgram ? 'Deepgram' : 'Groq'})` });
}

// ---------------------------------------------------------------------------
// IPC message handler
// ---------------------------------------------------------------------------
process.on('message', (msg) => {
  try {
    switch (msg.type) {
      case 'init': initialize(msg.config); break;
      case 'start': startRecording(); break;
      case 'stop': stopRecording(); break;
      case 'test': testConnection(); break;
      case 'status': process.send({ type: 'status', status: getStatus() }); break;
      case 'shutdown':
        stopRecording();
        log('info', 'Worker shutting down');
        setTimeout(() => process.exit(0), 500);
        break;
      default: log('warn', `Unknown message type: ${msg.type}`);
    }
  } catch (error) {
    log('error', `Error handling message ${msg.type}`, { error: error.message });
    process.send({ type: 'error', error: `Worker error: ${error.message}` });
  }
});

process.on('SIGTERM', () => {
  stopRecording();
  process.exit(0);
});
process.on('uncaughtException', (error) => {
  log('error', 'Uncaught exception in speech worker', { error: error.message });
});
process.on('unhandledRejection', (reason) => {
  log('error', 'Unhandled rejection in speech worker', { error: String(reason) });
});
log('info', 'Speech worker process started (Deepgram/Groq hybrid)', { pid: process.pid });
