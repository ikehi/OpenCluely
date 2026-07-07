const Groq = require('groq-sdk');
const logger = require('../core/logger').createServiceLogger('LLM');
const config = require('../core/config');
const { promptLoader } = require('../../prompt-loader');

class LLMService {
  constructor() {
    this.clients = [];
    this.currentClientIndex = 0;
    this.isInitialized = false;
    this.requestCount = 0;
    this.errorCount = 0;

    this.initializeClient();
  }

  initializeClient() {
    const apiKeyString = config.getApiKey('GROQ') || '';
    const apiKeys = apiKeyString.split(',').map(k => k.trim()).filter(k => k && k !== 'your-api-key-here' && k !== 'your_groq_api_key_here');

    if (apiKeys.length === 0) {
      logger.warn('Groq API key not configured', {
        keyExists: false
      });
      return;
    }

    try {
      this.clients = apiKeys.map(apiKey => new Groq({ apiKey }));
      this.currentClientIndex = 0;
      this.isInitialized = true;

      logger.info('Groq AI clients initialized successfully', {
        keyCount: apiKeys.length,
        model: config.get('llm.groq.model')
      });
    } catch (error) {
      logger.error('Failed to initialize Groq clients', {
        error: error.message
      });
    }
  }

  getGenerationConfig(overrides = {}) {
    const defaults = config.get('llm.groq.generation') || {};
    const fallback = {
      temperature: 0.4,
      max_tokens: 2048,
      top_p: 0.95
    };

    const merged = { ...fallback, ...defaults, ...overrides };
    return Object.fromEntries(
      Object.entries(merged).filter(([, value]) => value !== undefined && value !== null)
    );
  }

  /**
   * Process an image directly with Groq (using vision model)
   */
  async processImageWithSkill(imageBuffer, mimeType, activeSkill, sessionMemory = [], programmingLanguage = null) {
    if (!this.isInitialized) {
      throw new Error('LLM service not initialized. Check Groq API key configuration.');
    }

    if (!imageBuffer || !Buffer.isBuffer(imageBuffer)) {
      throw new Error('Invalid image buffer provided to processImageWithSkill');
    }

    const startTime = Date.now();
    this.requestCount++;

    try {
      const { promptLoader } = require('../../prompt-loader');
      const skillPrompt = promptLoader.getSkillPrompt(activeSkill, programmingLanguage) || '';

      const base64Image = imageBuffer.toString('base64');
      const imageUrl = `data:${mimeType};base64,${base64Image}`;
      
      const isCodingSkill = ['dsa', 'programming'].includes(activeSkill);

      if (isCodingSkill) {
        // --- STAGE 1: EXTRACTION (Vision Model) ---
        logger.info('[Pipeline Stage 1] Extracting text from single image via Vision Model');
        const extractionMessages = [{
          role: 'user',
          content: [
            { type: 'text', text: "You are an OCR and requirements-extraction expert. Extract all text, code snippets, constraints, and architectural requirements from this image verbatim. Do not attempt to solve the problem or write the final code. Just perfectly extract the raw requirements into text." },
            { type: 'image_url', image_url: { url: imageUrl } }
          ]
        }];
        
        // Use a small temp override for extraction tokens to ensure we don't blow the limit
        const sessionManager = require('../managers/session.manager');
        const originalMode = sessionManager.getResponseMode();
        sessionManager.setResponseMode('medium'); // forces vision output budget to ~1500 max
        
        let extractedText = '';
        try {
          extractedText = await this.executeRequest(extractionMessages, true);
        } finally {
          sessionManager.setResponseMode(originalMode);
        }
        
        // --- STAGE 2: GENERATION (Text Model) ---
        logger.info('[Pipeline Stage 2] Generating final code via Text Model', { extractedTextLength: extractedText.length });
        const generationPayload = `Please act as the Principal Staff Engineer and implement the following architectural requirements completely, following all system prompt rules. Do not just repeat the requirements; write the full code solution.\n\n=== EXTRACTED REQUIREMENTS ===\n${extractedText}\n==============================`;
        
        const finalResult = await this.processTextWithSkill(generationPayload, activeSkill, sessionMemory, programmingLanguage);
        finalResult.metadata.isImageAnalysis = true; // flag it so UI knows it started as an image
        return finalResult;
      }

      // Default single-stage execution for non-coding skills
      const messages = [];

      if (skillPrompt && skillPrompt.trim().length > 0) {
        messages.push({ role: 'system', content: skillPrompt });
      }

      messages.push({
        role: 'user',
        content: [
          { type: 'text', text: this.formatImageInstruction(activeSkill, programmingLanguage) },
          { type: 'image_url', image_url: { url: imageUrl } }
        ]
      });

      const responseText = await this.executeRequest(messages, true);

      const finalResponse = programmingLanguage
        ? this.enforceProgrammingLanguage(responseText, programmingLanguage)
        : responseText;

      logger.logPerformance('LLM image processing', startTime, {
        activeSkill,
        imageSize: imageBuffer.length,
        responseLength: finalResponse.length,
        programmingLanguage: programmingLanguage || 'not specified',
        requestId: this.requestCount
      });

      return {
        response: finalResponse,
        metadata: {
          skill: activeSkill,
          programmingLanguage,
          processingTime: Date.now() - startTime,
          requestId: this.requestCount,
          usedFallback: false,
          isImageAnalysis: true,
          mimeType
        }
      };
    } catch (error) {
      this.errorCount++;
      logger.error('LLM image processing failed', {
        error: error.message,
        activeSkill,
        requestId: this.requestCount
      });

      // Temporarily surface the raw error to the chat UI instead of the generic fallback
      return {
        response: `[Vision Model Error]: ${error.message}`,
        metadata: {
          skill: activeSkill,
          usedFallback: true,
          isImageAnalysis: true
        }
      };
    }
  }

  async processMultipleImagesWithSkill(images, activeSkill, sessionMemory = [], programmingLanguage = null) {
    if (!this.isInitialized) {
      throw new Error('LLM service not initialized. Check Groq API key configuration.');
    }

    if (!images || !Array.isArray(images) || images.length === 0) {
      throw new Error('Invalid images array provided to processMultipleImagesWithSkill');
    }

    const startTime = Date.now();
    this.requestCount++;

    try {
      const { promptLoader } = require('../../prompt-loader');
      const skillPrompt = promptLoader.getSkillPrompt(activeSkill, programmingLanguage) || '';

      const isCodingSkill = ['dsa', 'programming'].includes(activeSkill);

      if (isCodingSkill) {
        // --- STAGE 1: EXTRACTION (Vision Model) ---
        logger.info('[Pipeline Stage 1] Extracting text from multiple images via Vision Model');
        const contentArray = [
          { type: 'text', text: "You are an OCR and requirements-extraction expert. Extract all text, code snippets, constraints, and architectural requirements from these images verbatim. Do not attempt to solve the problem or write the final code. Just perfectly extract the raw requirements into text." }
        ];
        
        images.forEach((img) => {
          const base64Image = img.imageBuffer.toString('base64');
          const imageUrl = `data:${img.mimeType || 'image/png'};base64,${base64Image}`;
          contentArray.push({ type: 'image_url', image_url: { url: imageUrl } });
        });

        const extractionMessages = [{ role: 'user', content: contentArray }];
        
        // Use a small temp override for extraction tokens
        const sessionManager = require('../managers/session.manager');
        const originalMode = sessionManager.getResponseMode();
        sessionManager.setResponseMode('medium');
        
        let extractedText = '';
        try {
          extractedText = await this.executeRequest(extractionMessages, true);
        } finally {
          sessionManager.setResponseMode(originalMode);
        }
        
        // --- STAGE 2: GENERATION (Text Model) ---
        logger.info('[Pipeline Stage 2] Generating final code via Text Model', { extractedTextLength: extractedText.length });
        const generationPayload = `Please act as the Principal Staff Engineer and implement the following architectural requirements completely, following all system prompt rules. Do not just repeat the requirements; write the full code solution.\n\n=== EXTRACTED REQUIREMENTS ===\n${extractedText}\n==============================`;
        
        const finalResult = await this.processTextWithSkill(generationPayload, activeSkill, sessionMemory, programmingLanguage);
        finalResult.metadata.isImageAnalysis = true;
        finalResult.metadata.imageCount = images.length;
        return finalResult;
      }

      // Default single-stage execution for non-coding skills
      const messages = [];

      if (skillPrompt && skillPrompt.trim().length > 0) {
        messages.push({ role: 'system', content: skillPrompt });
      }

      const contentArray = [];
      const imageCount = images.length;
      
      const langNote = programmingLanguage ? ` Use only ${programmingLanguage.toUpperCase()} for any code.` : '';
      
      const instruction = `You are given ${imageCount} screenshots that together form a single question. Analyze them all together. CRITICAL: You must provide ONLY the direct final answer. If it's a coding question, output ONLY the code. If it's multiple choice, output ONLY the correct option. Do NOT restate the problem, do NOT provide a breakdown, do NOT provide explanations, and do NOT use templates like "Step 1:". Just the raw answer.${langNote}`;
      
      contentArray.push({ type: 'text', text: instruction });

      // Add each image to the content array
      images.forEach((img) => {
        const base64Image = img.imageBuffer.toString('base64');
        const imageUrl = `data:${img.mimeType || 'image/png'};base64,${base64Image}`;
        contentArray.push({ type: 'image_url', image_url: { url: imageUrl } });
      });

      messages.push({
        role: 'user',
        content: contentArray
      });

      const responseText = await this.executeRequest(messages, true);

      const finalResponse = programmingLanguage
        ? this.enforceProgrammingLanguage(responseText, programmingLanguage)
        : responseText;

      logger.logPerformance('LLM multi-image processing', startTime, {
        activeSkill,
        imageCount,
        responseLength: finalResponse.length,
        programmingLanguage: programmingLanguage || 'not specified',
        requestId: this.requestCount
      });

      return {
        response: finalResponse,
        metadata: {
          skill: activeSkill,
          programmingLanguage,
          processingTime: Date.now() - startTime,
          requestId: this.requestCount,
          usedFallback: false,
          isImageAnalysis: true,
          imageCount
        }
      };
    } catch (error) {
      this.errorCount++;
      logger.error('LLM multi-image processing failed', {
        error: error.message,
        activeSkill,
        requestId: this.requestCount
      });

      return {
        response: `[Vision Model Error (Batch)]: ${error.message}`,
        metadata: {
          skill: activeSkill,
          usedFallback: true,
          isImageAnalysis: true
        }
      };
    }
  }

  formatImageInstruction(activeSkill, programmingLanguage) {
    const langNote = programmingLanguage ? ` Use only ${programmingLanguage.toUpperCase()} for any code.` : '';
    
    return `Analyze this image. CRITICAL: You must provide ONLY the direct final answer. If it's a coding question, output ONLY the code. If it's multiple choice, output ONLY the correct option. Do NOT restate the problem, do NOT provide a breakdown, do NOT provide explanations, and do NOT use templates like "Step 1:". Just the raw answer.${langNote}`;
  }

  async processTextWithSkill(text, activeSkill, sessionMemory = [], programmingLanguage = null) {
    if (!this.isInitialized) {
      throw new Error('LLM service not initialized. Check Groq API key configuration.');
    }

    const startTime = Date.now();
    this.requestCount++;

    try {
      logger.info('Processing text with LLM', {
        activeSkill,
        textLength: text.length,
        hasSessionMemory: sessionMemory.length > 0,
        requestId: this.requestCount
      });

      const messages = this.buildGroqRequest(text, activeSkill, sessionMemory, programmingLanguage);
      const responseText = await this.executeRequestWithFallback(messages);

      const finalResponse = programmingLanguage
        ? this.enforceProgrammingLanguage(responseText, programmingLanguage)
        : responseText;

      logger.logPerformance('LLM text processing', startTime, {
        activeSkill,
        textLength: text.length,
        responseLength: finalResponse.length,
        requestId: this.requestCount
      });

      return {
        response: finalResponse,
        metadata: {
          skill: activeSkill,
          programmingLanguage,
          processingTime: Date.now() - startTime,
          requestId: this.requestCount,
          usedFallback: false
        }
      };
    } catch (error) {
      this.errorCount++;
      logger.error('LLM processing failed', {
        error: error.message,
        activeSkill,
        requestId: this.requestCount
      });

      if (config.get('llm.groq.fallbackEnabled')) {
        return this.generateFallbackResponse(text, activeSkill);
      }

      throw error;
    }
  }

  async processTranscriptionWithIntelligentResponse(text, activeSkill, sessionMemory = [], programmingLanguage = null) {
    if (!this.isInitialized) {
      throw new Error('LLM service not initialized. Check Groq API key configuration.');
    }

    const startTime = Date.now();
    this.requestCount++;

    try {
      logger.info('Processing transcription with intelligent response', {
        activeSkill,
        textLength: text.length,
        requestId: this.requestCount
      });

      const messages = this.buildIntelligentTranscriptionRequest(text, activeSkill, sessionMemory, programmingLanguage);
      const responseText = await this.executeRequestWithFallback(messages);

      const finalResponse = programmingLanguage
        ? this.enforceProgrammingLanguage(responseText, programmingLanguage)
        : responseText;

      return {
        response: finalResponse,
        metadata: {
          skill: activeSkill,
          programmingLanguage,
          processingTime: Date.now() - startTime,
          requestId: this.requestCount,
          usedFallback: false,
          isTranscriptionResponse: true
        }
      };
    } catch (error) {
      this.errorCount++;
      logger.error('LLM transcription processing failed', {
        error: error.message,
        activeSkill,
        requestId: this.requestCount
      });

      if (config.get('llm.groq.fallbackEnabled')) {
        return this.generateIntelligentFallbackResponse(text, activeSkill);
      }

      throw error;
    }
  }

  enforceProgrammingLanguage(text, programmingLanguage) {
    try {
      if (!text || !programmingLanguage) return text;
      const norm = String(programmingLanguage).toLowerCase();
      const fenceTagMap = { cpp: 'cpp', c: 'c', python: 'python', java: 'java', javascript: 'javascript', js: 'javascript' };
      const fenceTag = fenceTagMap[norm] || norm || 'text';

      const replacedBackticks = text.replace(/```([^\n]*)\n/g, (match, info) => {
        const current = (info || '').trim();
        if (current.split(/\s+/)[0].toLowerCase() === fenceTag) return match;
        return '```' + fenceTag + '\n';
      });

      const normalizedTildes = replacedBackticks.replace(/~~~([^\n]*)\n/g, () => '```' + fenceTag + '\n');
      return normalizedTildes;
    } catch (_) {
      return text;
    }
  }

  buildGroqRequest(text, activeSkill, sessionMemory, programmingLanguage) {
    const sessionManager = require('../managers/session.manager');

    const codingSkills = ['dsa', 'programming'];

    if (sessionManager && typeof sessionManager.getConversationHistory === 'function') {
      // For coding skills, skip history entirely — each question is self-contained
      // and past code responses are too large and push requests over the TPM limit.
      const conversationHistory = codingSkills.includes(activeSkill)
        ? []
        : sessionManager.getConversationHistory(4);
      const skillContext = sessionManager.getSkillContext(activeSkill, programmingLanguage);
      return this.buildGroqRequestWithHistory(text, activeSkill, conversationHistory, skillContext, programmingLanguage);
    }

    const requestComponents = promptLoader.getRequestComponents(
      activeSkill,
      text,
      sessionMemory,
      programmingLanguage
    );

    const messages = [];

    if (requestComponents.shouldUseModelMemory && requestComponents.skillPrompt) {
      messages.push({ role: 'system', content: requestComponents.skillPrompt });
    }

    messages.push({ role: 'user', content: this.formatUserMessage(text, activeSkill) });
    return messages;
  }

  buildGroqRequestWithHistory(text, activeSkill, conversationHistory, skillContext, programmingLanguage) {
    const messages = [];
    const sessionManager = require('../managers/session.manager');
    const documentContext = sessionManager.getDocumentContext();

    if (documentContext) {
      messages.push({ role: 'system', content: `## Reference Document Context\n${documentContext}\n\n## FIRST-PERSON RULE\nYou must adopt a first-person persona based on the reference document context provided above. When answering questions, speak directly from the perspective of the document's subject or author. Use "I", "me", "my". Keep your answers extremely concise (not too long, not too short). Do not break character.` });
    }

    if (skillContext.skillPrompt) {
      messages.push({ role: 'system', content: skillContext.skillPrompt });
    }

    const conversationContents = conversationHistory
      .filter(event => event.role !== 'system' && event.content && typeof event.content === 'string' && event.content.trim().length > 0)
      .map(event => ({
        role: event.role === 'model' ? 'assistant' : 'user',
        content: event.content.trim()
      }));

    messages.push(...conversationContents);

    const formattedMessage = this.formatUserMessage(text, activeSkill);
    if (!formattedMessage || formattedMessage.trim().length === 0) {
      throw new Error('Failed to format user message or message is empty');
    }

    messages.push({ role: 'user', content: formattedMessage });
    return messages;
  }

  buildIntelligentTranscriptionRequest(text, activeSkill, sessionMemory, programmingLanguage) {
    const cleanText = text && typeof text === 'string' ? text.trim() : '';
    if (!cleanText) {
      throw new Error('Empty or invalid transcription text provided');
    }

    const sessionManager = require('../managers/session.manager');

    const codingSkills = ['dsa', 'programming'];

    if (sessionManager && typeof sessionManager.getConversationHistory === 'function') {
      // For coding skills, skip history — each question is self-contained.
      // Previous code responses are too large and push requests over the TPM limit.
      const conversationHistory = codingSkills.includes(activeSkill)
        ? []
        : sessionManager.getConversationHistory(4);
      const skillContext = sessionManager.getSkillContext(activeSkill, programmingLanguage);
      return this.buildIntelligentTranscriptionRequestWithHistory(cleanText, activeSkill, conversationHistory, skillContext, programmingLanguage);
    }

    const messages = [];
    const documentContext = sessionManager ? sessionManager.getDocumentContext() : null;
    const intelligentPrompt = this.getIntelligentTranscriptionPrompt(activeSkill, programmingLanguage, documentContext);

    if (intelligentPrompt) {
      messages.push({ role: 'system', content: intelligentPrompt });
    }

    messages.push({ role: 'user', content: cleanText });
    return messages;
  }

  buildIntelligentTranscriptionRequestWithHistory(text, activeSkill, conversationHistory, skillContext, programmingLanguage) {
    const messages = [];
    const sessionManager = require('../managers/session.manager');
    const documentContext = sessionManager.getDocumentContext();
    const intelligentPrompt = this.getIntelligentTranscriptionPrompt(activeSkill, programmingLanguage, documentContext);

    if (intelligentPrompt) {
      messages.push({ role: 'system', content: intelligentPrompt });
    }

    const conversationContents = conversationHistory
      .filter(event => event.role !== 'system' && event.content && typeof event.content === 'string' && event.content.trim().length > 0)
      .slice(-4)
      .map(event => ({
        role: event.role === 'model' ? 'assistant' : 'user',
        content: event.content.trim()
      }));

    messages.push(...conversationContents);

    const cleanText = text && typeof text === 'string' ? text.trim() : '';
    if (!cleanText) {
      throw new Error('Empty or invalid transcription text provided');
    }

    messages.push({ role: 'user', content: cleanText });
    return messages;
  }

  getIntelligentTranscriptionPrompt(activeSkill, programmingLanguage, documentContext = null) {
    const sessionManager = require('../managers/session.manager');
    const mode = sessionManager.getResponseMode();

    // For coding-related skills, bypass the interview-whispering prompt entirely.
    // Instead, use the skill's own system prompt which enforces optimal code output.
    const codingSkills = ['dsa', 'programming'];
    if (codingSkills.includes(activeSkill)) {
      const skillPrompt = promptLoader.getSkillPrompt(activeSkill, programmingLanguage) || '';
      // Append a mode-aware instruction on top of the skill prompt
      let modeInstruction = '';
      if (mode === 'simple') {
        modeInstruction = '\n\nIMPORTANT: The user selected "Simple" mode — provide ONLY the final code with no explanation whatsoever.';
      } else if (mode === 'medium') {
        modeInstruction = '\n\nIMPORTANT: The user selected "Medium" mode — provide a one-line approach summary, then the complete code.';
      } else {
        modeInstruction = '\n\nIMPORTANT: The user selected "Complex" mode — briefly state the optimal time/space approach, then provide the complete runnable code, then a short complexity analysis.';
      }
      return skillPrompt + modeInstruction;
    }

    let prompt = `You are whispering answers to an interviewee during a live interview. The transcription has both the interviewer and interviewee's voice — ONLY answer the interviewer's questions. Ignore anything the interviewee says.\n\nABSOLUTE RULES:`;

    if (mode === 'simple') {
      prompt += `\n- Answer in 1-2 sentences only. Be extremely concise.\n- NEVER use bullet points, numbered lists, headers, bold, markdown, or any formatting. Only plain flowing sentences.\n- NEVER include code.\n- Sound like a confident person speaking casually.`;
    } else if (mode === 'medium') {
      prompt += `\n- Your answer must be 1-2 short paragraphs.\n- NEVER use bullet points, numbered lists, headers, bold, markdown, or any formatting. Only plain flowing sentences.\n- NEVER include code.\n- Sound like a confident person speaking casually.`;
    } else {
      prompt += `\n- Your answer must be 2-3 SHORT paragraphs. Each paragraph is 2 sentences max. No exceptions, even for complex questions.\n- NEVER use bullet points, numbered lists, headers, bold, markdown, or any formatting. Only plain flowing sentences.\n- NEVER give each sub-topic its own paragraph. Blend everything together tightly.\n- NEVER include code.\n- Sound like a confident person speaking casually — use filler words like "so", "actually", "you know", "honestly" naturally. Do not sound like a textbook.\n- For simple questions (naming, listing, yes/no), answer in 1-2 sentences only.`;
    }

    if (documentContext) {
      prompt += `\n\nSpeak as the person described below. Use "I", "me", "my". Stay in character.\n\n${documentContext}`;
    }

    return prompt;
  }

  formatUserMessage(text, activeSkill) {
    return `Context: ${activeSkill.toUpperCase()} analysis request\n\nText to analyze:\n${text}`;
  }

  async executeRequest(messages, isVision = false) {
    const sessionManager = require('../managers/session.manager');
    const mode = sessionManager.getResponseMode();
    let maxTokens = 2500;
    if (mode === 'simple') maxTokens = 450;
    if (mode === 'medium') maxTokens = 850;

    const activeSkill = sessionManager.currentSkill;
    const isCodingSkill = ['dsa', 'programming'].includes(activeSkill);

    // Fast model rotation pool — each model has independent rate limits on Groq free tier
    let modelPool = isVision
      ? ['meta-llama/llama-4-scout-17b-16e-instruct']
      : ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'llama3-8b-8192', 'gemma2-9b-it'];

    // Enforce specific high-tier models for coding tasks to ensure senior-level quality
    if (isCodingSkill) {
      if (isVision) {
        // qwen3.6-27b natively supports vision on Groq, making it perfect for coding screenshots
        modelPool = ['qwen/qwen3.6-27b'];
      } else {
        modelPool = [
          'openai/gpt-oss-120b',
          'qwen/qwen3.6-27b',
          'openai/gpt-oss-20b'
        ];
      }
      // TPM limit for these models is 8000 (input + output combined).
      // If vision is used, images consume ~3000-5000 tokens. We must drastically lower maxTokens to fit the 8000 limit.
      if (isVision) {
        if (mode === 'complex') maxTokens = 2500;
        else if (mode === 'medium') maxTokens = 1500;
      } else {
        // Standard text-only coding request
        if (mode === 'complex') maxTokens = 6000;
        else if (mode === 'medium') maxTokens = 3500;
      }
      // simple stays at 450
    }

    const payload = {
      messages,
      model: modelPool[0],
      ...this.getGenerationConfig(),
      max_tokens: maxTokens
    };

    let lastError = null;

    // Round-robin: grab the next key instantly for this specific request.
    // This ensures concurrent requests process in parallel using completely different API keys, preventing bottlenecks.
    const requestStartingKeyIndex = this.currentClientIndex;
    if (this.clients.length > 0) {
      this.currentClientIndex = (this.currentClientIndex + 1) % this.clients.length;
    }

    // Try each model instantly on rate limit — zero delay rotation
    for (let i = 0; i < modelPool.length; i++) {
      payload.model = modelPool[i];

      // Try each API key for the current model
      for (let j = 0; j < this.clients.length; j++) {
        const clientIndex = (requestStartingKeyIndex + j) % this.clients.length;
        const currentClient = this.clients[clientIndex];

        try {
          const response = await currentClient.chat.completions.create(payload);

          if (!response.choices || response.choices.length === 0) {
            throw new Error('Empty response from Groq API');
          }

          return response.choices[0].message.content;
        } catch (error) {
          lastError = error;
          const errorInfo = this.analyzeError(error);

          logger.warn(`Groq model ${payload.model} failed on API key index ${clientIndex}`, {
            error: error.message,
            errorType: errorInfo.type,
            model: payload.model,
            keyIndex: clientIndex
          });

          // If rate limited OR request too large (TPM), try next key — each key has an independent bucket
          if (errorInfo.type === 'RATE_LIMIT_ERROR' || errorInfo.type === 'REQUEST_TOO_LARGE_ERROR') {
            continue;
          }

          // If auth error (e.g. invalid key), instantly try next key
          if (errorInfo.type === 'AUTH_ERROR') {
            continue;
          }

          // For other errors (like model decommissioned), break inner loop to move to next model
          break;
        }
      }

      // If we got here, all keys for this model failed. 
      if (lastError) {
        const errorInfo = this.analyzeError(lastError);

        // If rate limited or too large across all keys, switch model instantly
        if ((errorInfo.type === 'RATE_LIMIT_ERROR' || errorInfo.type === 'REQUEST_TOO_LARGE_ERROR') && i < modelPool.length - 1) {
          logger.info(`All keys exhausted for ${payload.model} (${errorInfo.type}), instantly switching to ${modelPool[i + 1]}`);
          continue;
        }

        // If it's the last model, we throw
        if (i === modelPool.length - 1) {
          throw new Error(`All Groq models and keys exhausted: ${lastError.message}`);
        }

        // Small delay for network errors etc
        if (errorInfo.type !== 'RATE_LIMIT_ERROR') {
          const delay = 1000 + Math.random() * 500;
          await this.delay(delay);
        }
      }
    }

    throw new Error(`All Groq models and keys exhausted. Last error: ${lastError ? lastError.message : 'Unknown'}`);
  }

  // 413 fallback — if ALL preferred coding models rejected the request because it was too
  // large, retry once with the standard model pool which has a much higher context limit.
  async executeRequestWithFallback(messages, isVision = false) {
    const sessionManager = require('../managers/session.manager');
    const activeSkill = sessionManager.currentSkill;
    const isCodingSkill = ['dsa', 'programming'].includes(activeSkill) && !isVision;

    try {
      return await this.executeRequest(messages, isVision);
    } catch (error) {
      // Only fall back when in coding mode AND the failure was a 413 (request too large)
      const is413 = error.message.includes('413') || error.message.includes('Request too large') || error.message.includes('reduce your message size');
      if (isCodingSkill && is413) {
        logger.warn('Coding model pool rejected request as too large — falling back to standard model pool', {
          activeSkill,
          errorPreview: error.message.substring(0, 120)
        });
        // Temporarily override currentSkill so executeRequest picks the standard pool
        const originalSkill = sessionManager.currentSkill;
        sessionManager.currentSkill = '_fallback';
        try {
          return await this.executeRequest(messages, isVision);
        } finally {
          sessionManager.currentSkill = originalSkill;
        }
      }
      throw error;
    }
  }

  async performPreflightCheck() {
    try {
      await this.testNetworkConnection({
        host: 'api.groq.com',
        port: 443,
        name: 'Groq API Endpoint'
      });
    } catch (error) {
      logger.warn('Preflight check failed', { error: error.message });
    }
  }

  analyzeError(error) {
    const errorMessage = error.message.toLowerCase();

    if (errorMessage.includes('fetch failed') || errorMessage.includes('network error') || errorMessage.includes('timeout')) {
      return { type: 'NETWORK_ERROR', isNetworkError: true };
    }

    if (errorMessage.includes('unauthorized') || errorMessage.includes('invalid api key')) {
      return { type: 'AUTH_ERROR', isNetworkError: false };
    }

    if (errorMessage.includes('rate limit') || errorMessage.includes('too many requests') || errorMessage.includes('429')) {
      return { type: 'RATE_LIMIT_ERROR', isNetworkError: false };
    }

    if (errorMessage.includes('413') || errorMessage.includes('request too large') || errorMessage.includes('reduce your message size')) {
      return { type: 'REQUEST_TOO_LARGE_ERROR', isNetworkError: false };
    }

    return { type: 'UNKNOWN_ERROR', isNetworkError: false };
  }

  async checkNetworkConnectivity() {
    const connectivityTests = [
      { host: 'google.com', port: 443, name: 'Google (HTTPS)' },
      { host: 'api.groq.com', port: 443, name: 'Groq API Endpoint' }
    ];

    const results = await Promise.allSettled(
      connectivityTests.map(test => this.testNetworkConnection(test))
    );

    const connectivity = {
      timestamp: new Date().toISOString(),
      tests: results.map((result, index) => ({
        ...connectivityTests[index],
        success: result.status === 'fulfilled' && result.value,
        error: result.status === 'rejected' ? result.reason.message : null
      }))
    };

    return connectivity;
  }

  async testNetworkConnection({ host, port, name }) {
    return new Promise((resolve, reject) => {
      const net = require('net');
      const socket = new net.Socket();

      const timeout = setTimeout(() => {
        socket.destroy();
        reject(new Error(`Connection timeout to ${host}:${port}`));
      }, 5000);

      socket.on('connect', () => {
        clearTimeout(timeout);
        socket.destroy();
        resolve(true);
      });

      socket.on('error', (error) => {
        clearTimeout(timeout);
        reject(new Error(`Connection failed to ${host}:${port}: ${error.message}`));
      });

      socket.connect(port, host);
    });
  }

  generateFallbackResponse(text, activeSkill) {
    const fallbackResponses = {
      'dsa': 'This appears to be a data structures and algorithms problem. Consider breaking it down into smaller components and identifying the appropriate algorithm or data structure to use.',
      'system-design': 'For this system design question, consider scalability, reliability, and the trade-offs between different architectural approaches.',
      'programming': 'This looks like a programming challenge. Focus on understanding the requirements, edge cases, and optimal time/space complexity.',
      'default': 'I can help analyze this content. Please ensure your Groq API key is properly configured for detailed analysis.'
    };

    const response = fallbackResponses[activeSkill] || fallbackResponses.default;

    return {
      response,
      metadata: {
        skill: activeSkill,
        processingTime: 0,
        requestId: this.requestCount,
        usedFallback: true
      }
    };
  }

  generateIntelligentFallbackResponse(text, activeSkill) {
    const response = `Yeah, I'm listening. Ask your question relevant to ${activeSkill}.`;
    return {
      response,
      metadata: {
        skill: activeSkill,
        processingTime: 0,
        requestId: this.requestCount,
        usedFallback: true,
        isTranscriptionResponse: true
      }
    };
  }

  async testConnection() {
    if (!this.isInitialized || !this.clients || this.clients.length === 0) {
      return { success: false, error: 'Service not initialized' };
    }

    try {
      const startTime = Date.now();
      const currentClient = this.clients[this.currentClientIndex];
      const response = await currentClient.chat.completions.create({
        messages: [{ role: 'user', content: 'Test connection. Please respond with "OK".' }],
        model: config.get('llm.groq.model') || 'llama-3.3-70b-versatile',
        max_tokens: 10
      });
      const latency = Date.now() - startTime;

      return {
        success: true,
        response: response.choices[0].message.content,
        latency
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  updateApiKey(newApiKey) {
    process.env.GROQ_API_KEY = newApiKey;
    this.isInitialized = false;
    this.initializeClient();
  }

  getStats() {
    return {
      isInitialized: this.isInitialized,
      requestCount: this.requestCount,
      errorCount: this.errorCount,
      successRate: this.requestCount > 0 ? ((this.requestCount - this.errorCount) / this.requestCount) * 100 : 0,
      config: config.get('llm.groq')
    };
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

module.exports = new LLMService();