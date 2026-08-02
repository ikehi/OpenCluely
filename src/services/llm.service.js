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
    this.visionModel = 'qwen/qwen3.6-27b';

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
        model: config.get('llm.groq.model'),
        keyRotation: 'round-robin across all keys on failure'
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
      const usesTwoStageImagePipeline = isCodingSkill || activeSkill === 'general';

      if (usesTwoStageImagePipeline) {
        const sessionManager = require('../managers/session.manager');
        const originalMode = sessionManager.getResponseMode();
        let screenshotType = 'other';

        if (!isCodingSkill) {
          screenshotType = await this.classifyScreenshotType(imageUrl, activeSkill);
          logger.info('[Pipeline] Screenshot classified', { screenshotType });
          if (screenshotType === 'visual_puzzle') {
            logger.info('[Pipeline] Fast path: solving visual puzzle directly (skip extraction)');
            const reasoningResponse = await this.answerReasoningMcqFromImage(imageUrl, activeSkill);
            return {
              response: reasoningResponse,
              metadata: {
                skill: activeSkill,
                programmingLanguage,
                processingTime: Date.now() - startTime,
                requestId: this.requestCount,
                usedFallback: false,
                isImageAnalysis: true,
                isMcq: true,
                mcqType: 'reasoning',
                mimeType
              }
            };
          }
        }

        // --- STAGE 1: EXTRACTION (Qwen Vision) ---
        logger.info('[Pipeline Stage 1] Extracting content from single image via vision', { activeSkill });
        const extractionPrompt = isCodingSkill
          ? 'You are an OCR and requirements-extraction expert. Extract all text, code snippets, constraints, and architectural requirements from this image verbatim. Do not attempt to solve the problem or write the final code. Just perfectly extract the raw requirements into text.'
          : this.getGeneralImageExtractionPrompt(1);

        const extractionMessages = [{
          role: 'user',
          content: [
            { type: 'text', text: extractionPrompt },
            { type: 'image_url', image_url: { url: imageUrl } }
          ]
        }];
        
        // Use a small temp override for extraction tokens to ensure we don't blow the limit
        sessionManager.setResponseMode('medium'); // forces vision output budget to ~1500 max
        
        let extractedText = '';
        try {
          extractedText = await this.executeRequest(extractionMessages, true, activeSkill, {
            lowTemperature: true,
            maxTokensOverride: (screenshotType === 'text_mcq' || screenshotType === 'math_mcq') ? 2200 : 900,
            noThinkingSalvage: true
          });
        } finally {
          sessionManager.setResponseMode(originalMode);
        }

        const rawExtractedText = extractedText;

        if (!isCodingSkill) {
          extractedText = this.sanitizeExtractedContent(extractedText);
          extractedText = this.stripModelThinking(extractedText);
          if (!extractedText.trim() && rawExtractedText.trim()) {
            extractedText = this.salvageTextFromThinking(rawExtractedText);
            extractedText = this.sanitizeExtractedContent(extractedText);
          }
          if (this.isMultipleChoiceContent(extractedText)) {
            extractedText = this.sanitizeMcqExtraction(extractedText);
          }
        }

        const reasoningResult = !isCodingSkill
          ? await this.tryReasoningMcqVisionPath(
            extractedText,
            imageUrl,
            activeSkill,
            startTime,
            programmingLanguage,
            mimeType,
            rawExtractedText,
            screenshotType
          )
          : null;
        if (reasoningResult) {
          return reasoningResult;
        }

        if (!isCodingSkill && extractedText.trim().length < 40) {
          throw new Error('Could not read screenshot content. Try centering the question and screenshot again.');
        }
        
        // --- STAGE 2: GENERATION (Text Model) ---
        logger.info('[Pipeline Stage 2] Generating final answer via Text Model', { activeSkill, extractedTextLength: extractedText.length });
        let generationPayload;
        let effectiveProgrammingLanguage = programmingLanguage;
        let requestOptions;

        if (isCodingSkill) {
          generationPayload = `Please act as the Principal Staff Engineer and implement the following architectural requirements completely, following all system prompt rules. Do not just repeat the requirements; write the full code solution.\n\n=== EXTRACTED REQUIREMENTS ===\n${extractedText}\n==============================`;
        } else {
          const generalGeneration = this.buildGeneralImageGenerationPayload(extractedText, { screenshotType });
          generationPayload = generalGeneration.payload;
          if (!effectiveProgrammingLanguage && generalGeneration.detectedLang) {
            effectiveProgrammingLanguage = generalGeneration.detectedLang;
          }

          requestOptions = {};
          if (generalGeneration.isCodingExercise) {
            requestOptions.preferCodingTokenBudget = true;
          }
          if (generalGeneration.isMcq) {
            requestOptions.mcqAnswerOnly = true;
            requestOptions.mcqType = generalGeneration.mcqType;
            requestOptions.mcqQuestionCount = generalGeneration.questionCount;
            requestOptions.mcqExtractedText = extractedText;
            logger.info('[Pipeline Stage 2] Solving MCQ via text model', {
              mcqType: generalGeneration.mcqType,
              questionCount: generalGeneration.questionCount,
              screenshotType,
              extractedPreview: extractedText.substring(0, 200)
            });
          }
          if (Object.keys(requestOptions).length === 0) {
            requestOptions = undefined;
          }
        }

        const stageTwoSessionMemory = isCodingSkill ||
          (!isCodingSkill && (
            this.isCodingExerciseContent(extractedText) ||
            this.isMultipleChoiceContent(extractedText) ||
            screenshotType === 'text_mcq' ||
            screenshotType === 'math_mcq'
          ))
          ? []
          : sessionMemory;

        const finalResult = await this.processTextWithSkill(
          generationPayload,
          activeSkill,
          stageTwoSessionMemory,
          effectiveProgrammingLanguage,
          requestOptions
        );
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
      const usesTwoStageImagePipeline = isCodingSkill || activeSkill === 'general';

      if (usesTwoStageImagePipeline) {
        const sessionManager = require('../managers/session.manager');
        const originalMode = sessionManager.getResponseMode();
        const firstImageUrl = this.getImageUrlFromSource(images);
        let screenshotType = 'other';

        if (!isCodingSkill && firstImageUrl) {
          screenshotType = await this.classifyScreenshotType(firstImageUrl, activeSkill);
          logger.info('[Pipeline] Screenshot classified', { screenshotType, imageCount: images.length });
          if (screenshotType === 'visual_puzzle') {
            logger.info('[Pipeline] Fast path: solving visual puzzle directly (skip extraction)', { imageCount: images.length });
            const reasoningResponse = await this.answerReasoningMcqFromImage(images, activeSkill);
            return {
              response: reasoningResponse,
              metadata: {
                skill: activeSkill,
                programmingLanguage,
                processingTime: Date.now() - startTime,
                requestId: this.requestCount,
                usedFallback: false,
                isImageAnalysis: true,
                isMcq: true,
                mcqType: 'reasoning',
                imageCount: images.length
              }
            };
          }
        }

        // --- STAGE 1: EXTRACTION (Qwen Vision) ---
        logger.info('[Pipeline Stage 1] Extracting content from multiple images via vision', { activeSkill });
        const extractionPrompt = isCodingSkill
          ? 'You are an OCR and requirements-extraction expert. Extract all text, code snippets, constraints, and architectural requirements from these images verbatim. Do not attempt to solve the problem or write the final code. Just perfectly extract the raw requirements into text.'
          : this.getGeneralImageExtractionPrompt(images.length);

        const contentArray = [{ type: 'text', text: extractionPrompt }];
        
        images.forEach((img) => {
          const base64Image = img.imageBuffer.toString('base64');
          const imageUrl = `data:${img.mimeType || 'image/png'};base64,${base64Image}`;
          contentArray.push({ type: 'image_url', image_url: { url: imageUrl } });
        });

        const extractionMessages = [{ role: 'user', content: contentArray }];
        
        sessionManager.setResponseMode('medium');
        
        let extractedText = '';
        try {
          extractedText = await this.executeRequest(extractionMessages, true, activeSkill, {
            lowTemperature: true,
            maxTokensOverride: (screenshotType === 'text_mcq' || screenshotType === 'math_mcq') ? 2200 : 900,
            noThinkingSalvage: true
          });
        } finally {
          sessionManager.setResponseMode(originalMode);
        }

        const rawExtractedText = extractedText;

        if (!isCodingSkill) {
          extractedText = this.sanitizeExtractedContent(extractedText);
          extractedText = this.stripModelThinking(extractedText);
          if (!extractedText.trim() && rawExtractedText.trim()) {
            extractedText = this.salvageTextFromThinking(rawExtractedText);
            extractedText = this.sanitizeExtractedContent(extractedText);
          }
          if (this.isMultipleChoiceContent(extractedText)) {
            extractedText = this.sanitizeMcqExtraction(extractedText);
          }
        }

        const reasoningResult = !isCodingSkill
          ? await this.tryReasoningMcqVisionPath(
            extractedText,
            images,
            activeSkill,
            startTime,
            programmingLanguage,
            null,
            rawExtractedText,
            screenshotType
          )
          : null;
        if (reasoningResult) {
          return reasoningResult;
        }

        if (!isCodingSkill && extractedText.trim().length < 40) {
          throw new Error('Could not read screenshot content. Try centering the question and screenshot again.');
        }
        
        // --- STAGE 2: GENERATION (Text Model) ---
        logger.info('[Pipeline Stage 2] Generating final answer via Text Model', { activeSkill, extractedTextLength: extractedText.length });
        let generationPayload;
        let effectiveProgrammingLanguage = programmingLanguage;
        let requestOptions;

        if (isCodingSkill) {
          generationPayload = `Please act as the Principal Staff Engineer and implement the following architectural requirements completely, following all system prompt rules. Do not just repeat the requirements; write the full code solution.\n\n=== EXTRACTED REQUIREMENTS ===\n${extractedText}\n==============================`;
        } else {
          const generalGeneration = this.buildGeneralImageGenerationPayload(extractedText, { screenshotType });
          generationPayload = generalGeneration.payload;
          if (!effectiveProgrammingLanguage && generalGeneration.detectedLang) {
            effectiveProgrammingLanguage = generalGeneration.detectedLang;
          }

          requestOptions = {};
          if (generalGeneration.isCodingExercise) {
            requestOptions.preferCodingTokenBudget = true;
          }
          if (generalGeneration.isMcq) {
            requestOptions.mcqAnswerOnly = true;
            requestOptions.mcqType = generalGeneration.mcqType;
            requestOptions.mcqQuestionCount = generalGeneration.questionCount;
            requestOptions.mcqExtractedText = extractedText;
            logger.info('[Pipeline Stage 2] Solving MCQ via text model', {
              mcqType: generalGeneration.mcqType,
              questionCount: generalGeneration.questionCount,
              screenshotType,
              imageCount: images.length,
              extractedPreview: extractedText.substring(0, 200)
            });
          }
          if (Object.keys(requestOptions).length === 0) {
            requestOptions = undefined;
          }
        }

        const stageTwoSessionMemory = isCodingSkill ||
          (!isCodingSkill && (
            this.isCodingExerciseContent(extractedText) ||
            this.isMultipleChoiceContent(extractedText) ||
            screenshotType === 'text_mcq' ||
            screenshotType === 'math_mcq'
          ))
          ? []
          : sessionMemory;

        const finalResult = await this.processTextWithSkill(
          generationPayload,
          activeSkill,
          stageTwoSessionMemory,
          effectiveProgrammingLanguage,
          requestOptions
        );
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

  async classifyScreenshotType(imageUrl, activeSkill) {
    try {
      const response = await this.executeRequest([{
        role: 'user',
        content: [
          {
            type: 'text',
            text: 'Look at this image. Reply with exactly ONE label only:\ntext_mcq — webpage/quiz with A B C D letter options and question text\nvisual_puzzle — shape/grid/matrix pattern with numbered shape options (1)(2)(3)(4)\nmath_mcq — math problems with numbers\nother'
          },
          { type: 'image_url', image_url: { url: imageUrl } }
        ]
      }], true, activeSkill, {
        forceModel: this.visionModel,
        maxTokensOverride: 24,
        lowTemperature: true
      });

      const label = (response || '').toLowerCase().trim();
      if (label.includes('text_mcq')) return 'text_mcq';
      if (label.includes('visual_puzzle')) return 'visual_puzzle';
      if (label.includes('math_mcq')) return 'math_mcq';
      if (label.includes('coding')) return 'coding';
      return 'other';
    } catch (error) {
      logger.warn('Screenshot classify failed', { error: error.message });
      return 'other';
    }
  }

  getImageUrlFromSource(imageSource) {
    if (typeof imageSource === 'string') return imageSource;
    if (Array.isArray(imageSource) && imageSource.length > 0) {
      const img = imageSource[0];
      if (typeof img === 'string') return img;
      const base64Image = img.imageBuffer.toString('base64');
      return `data:${img.mimeType || 'image/png'};base64,${base64Image}`;
    }
    return null;
  }

  getGeneralImageExtractionPrompt(imageCount = 1) {
    const imageLabel = imageCount === 1 ? 'this screenshot' : 'these screenshots';
    return `Extract ONLY the substantive problem content from ${imageLabel}.

INCLUDE: questions, exercises, requirements, answer choices (A/B/C/D), code snippets, constraints, timers, and the specified programming language.

For multiple choice quizzes: extract each question with ALL four options labeled A, B, C, D exactly as written. Preserve exact option text and order. Do not skip, merge, or invent options.

If questions use sub-labels like 1a), 1b), 1c) — preserve those labels exactly.
If options appear as bullet points without letters, label them A, B, C, D in order under each question.
If questions are numbered 1., 2., 3. — preserve those numbers.

IGNORE COMPLETELY (do not output any of these):
- Browser chrome, tabs, address bars, search bars
- Image viewer overlays: copyright notices, "Share", "Save", "Learn more", dimensions like "800 x 518", "Visit >", "using AI"
- App chat overlays: "Live Transcription & Chat", numbered answer lists like "1. A", dropdown labels like "General" or "Complex", timestamps
- Duplicate repeated blocks of text
- Unrelated URLs or page titles unless they are part of the question itself
- Watermarks and platform branding (e.g. "Powered by micro1") unless needed for context

Output clean, deduplicated text of the actual task only. Do NOT solve, answer, or write code.

IMPORTANT: Output plain extracted text only. Do not use thinking blocks, XML tags, or internal reasoning.`;
  }

  sanitizeExtractedContent(text) {
    if (!text || typeof text !== 'string') return '';

    const noisePatterns = [
      /^images may be subject to copyright\.?$/i,
      /^learn more$/i,
      /^share$/i,
      /^save$/i,
      /^using ai$/i,
      /^visit\s*>?$/i,
      /^\d+\s*x\s*\d+$/,
      /^https?:\/\/\S+$/i,
      /^micro1 interview questions and what to expect$/i,
      /^\d+[\.\)]\s*\*?\*?analyze the image/i,
      /^\d+[\.\)]\s*\*?\*?identify the relevant/i,
      /^\*\*analyze the image/i,
      /^\*\*identify the relevant/i,
      /^there's a header:/i
    ];

    const seen = new Set();
    const cleaned = [];

    for (const rawLine of text.split('\n')) {
      const line = rawLine.trim();
      if (!line) continue;
      if (noisePatterns.some((pattern) => pattern.test(line))) continue;

      const normalized = line.toLowerCase();
      if (seen.has(normalized)) continue;
      seen.add(normalized);
      cleaned.push(line);
    }

    return cleaned.join('\n');
  }

  sanitizeMcqExtraction(text) {
    if (!text || typeof text !== 'string') return '';

    const appUiPatterns = [
      /live transcription/i,
      /auto-detect lang/i,
      /^general$/i,
      /^complex$/i,
      /^medium$/i,
      /^simple$/i,
      /^programming$/i,
      /^dsa$/i,
      /^response$/i,
      /^screenshot/i,
      /opencluely/i
    ];

    const cleaned = [];

    for (const rawLine of text.split('\n')) {
      const line = rawLine.trim();
      if (!line) continue;
      if (/^\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?$/i.test(line)) continue;
      if (/^\d+[\.\)]\s*[A-D]$/i.test(line)) continue;
      if (appUiPatterns.some((pattern) => pattern.test(line))) continue;
      cleaned.push(line);
    }

    return cleaned.join('\n');
  }

  detectProgrammingLanguageFromText(text) {
    if (!text) return null;

    const patterns = [
      { lang: 'javascript', regex: /language:\s*javascript|write your javascript|\/\/\s*write your javascript/i },
      { lang: 'python', regex: /language:\s*python|write your python|#\s*write your python/i },
      { lang: 'java', regex: /language:\s*java|write your java/i },
      { lang: 'cpp', regex: /language:\s*c\+\+|write your c\+\+/i },
      { lang: 'c', regex: /language:\s*c\b|write your c solution/i }
    ];

    for (const { lang, regex } of patterns) {
      if (regex.test(text)) return lang;
    }

    return null;
  }

  isCodingExerciseContent(text) {
    if (!text) return false;

    return /coding exercise|frontend requirements|backend requirements|full[- ]stack|write your (javascript|python|java|c\+\+|c\b)|leetcode|hackerrank|codility|micro1|\/\/\s*write your|REST API|POST and GET routes/i.test(text);
  }

  isMultipleChoiceContent(text) {
    if (!text) return false;

    const letterOptions = (text.match(/\b[A-Da-d][\.\):]\s*\S/gi) || []).length;
    const parenLetterOptions = (text.match(/\([A-Da-d]\)/g) || []).length;
    const numberedOptions = (text.match(/\(\d\)/g) || []).length;
    const questionMarks = (text.match(/\?/g) || []).length;
    const hasGridPuzzle = questionMarks > 0 && /grid|matrix|row|column|figure|pattern/i.test(text);
    const quizKeywords = /multiple choice|select (the )?correct|choose (the )?correct|mcq|quiz/i.test(text);

    return quizKeywords ||
      letterOptions >= 4 ||
      parenLetterOptions >= 4 ||
      numberedOptions >= 4 ||
      hasGridPuzzle ||
      (questionMarks >= 2 && (letterOptions >= 2 || parenLetterOptions >= 2));
  }

  countMcqQuestions(text) {
    if (!text) return 1;

    const subPartHeaders = text.match(/(?:^|\n)\s*\d+[a-e][\.\)]/gi);
    const bracketSubParts = text.match(/(?:^|\n)\s*\[[a-e]\]/gi);
    if (subPartHeaders && subPartHeaders.length >= 2) {
      return subPartHeaders.length;
    }
    if (bracketSubParts && bracketSubParts.length >= 2) {
      return bracketSubParts.length;
    }

    const numberedListNums = new Set();
    for (const match of text.matchAll(/(?:^|\n)\s*(\d{1,2})\.\s+(?!\d)/g)) {
      numberedListNums.add(parseInt(match[1], 10));
    }
    if (numberedListNums.size >= 2) {
      return numberedListNums.size;
    }

    const qStyleHeaders = text.match(/(?:^|\n)\s*Q\d+[\.\):]/gi);
    if (qStyleHeaders && qStyleHeaders.length > 0) {
      return qStyleHeaders.length;
    }

    const numberedQuestionHeaders = text.match(
      /(?:^|\n)\s*(?:Q\d+[\.\):]|\d+[\.\)])\s+(?:Which|What|Is |Are |All |Mathematical|State |Below|Every |Neither|Having|Good |Self |If |How |When |Where |Why |Select|Choose|The )/gi
    );
    if (numberedQuestionHeaders && numberedQuestionHeaders.length > 0) {
      return numberedQuestionHeaders.length;
    }

    const optionMarkers = (text.match(/(?:^|\n)\s*[a-d][\.\)]\s+/gi) || []).length;
    const numberedOptionMarkers = (text.match(/\(\d\)/g) || []).length;
    const optionBasedCount = optionMarkers >= 4
      ? Math.round(optionMarkers / 4)
      : (numberedOptionMarkers >= 4 ? 1 : 0);

    const headerMatches = text.match(
      /(?:^|\n)\s*(?:Q\d+[\.\):]|\d{1,2}[\.\)])\s+(?:Which|What|Solve|Consider|Having|Good|Self|If|How|When|Where|Why|Select|Choose|The |Is |Are |Mathematical|State )/gi
    );
    const headerCount = headerMatches ? headerMatches.length : 0;

    if (headerCount > 0 && optionBasedCount > 0) {
      if (Math.abs(headerCount - optionBasedCount) <= 1) {
        return headerCount;
      }
      return Math.min(headerCount, optionBasedCount);
    }

    if (headerCount > 0) return headerCount;
    if (optionBasedCount > 0) return optionBasedCount;

    const questionLines = text.split('\n').filter((line) => {
      const trimmed = line.trim();
      return trimmed.endsWith('?') && !/^[a-d][\.\)]/i.test(trimmed);
    });
    if (questionLines.length > 0) {
      return questionLines.length;
    }

    return 1;
  }

  extractMcqAnswerLines(text) {
    return (text || '')
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => /^\d+[a-e]?[\.\):\-]\s+\S/i.test(line) || /^[A-D]\s*[—\-–:]\s*\S/i.test(line));
  }

  hasSubstantiveMcqAnswers(text, expectedCount = null) {
    const expanded = this.expandInlineMcqAnswers(text);
    const lines = this.extractMcqAnswerLines(expanded);
    if (lines.length === 0) return false;
    const minRequired = expectedCount ? Math.max(1, expectedCount - 1) : 1;
    return lines.length >= minRequired;
  }

  scoreMcqResponse(text, mcqType, expectedCount = null) {
    const expanded = this.expandInlineMcqAnswers(text);
    const lines = this.extractMcqAnswerLines(expanded);
    if (lines.length === 0) return 0;

    let score = lines.length * 10;
    if (expectedCount) {
      if (lines.length >= expectedCount) score += 25;
      else if (lines.length >= expectedCount - 1) score += 10;
    }

    for (const line of lines) {
      if (/^\d+[\.\):\-]\s+[A-D]\b.*[—\-–:]/i.test(line)) score += 8;
      else if (line.length > 12) score += 5;
      if (mcqType === 'math' && /[\d$¼¾½⅓⅔]/.test(line)) score += 4;
    }

    const letters = this.extractMcqAnswerLetters(text);
    if (letters.length >= 3 && this.isPatternMcqGarbage(text, expectedCount, true)) {
      score -= 40;
    }

    return score;
  }

  formatMathMcqAnswer(text, expectedCount = null) {
    const lines = (text || '').split('\n').map((line) => line.trim()).filter(Boolean);
    const out = [];
    let autoNum = 1;

    for (const line of lines) {
      let match = line.match(/^(\d+)[\.\):\-]\s+([A-D])\s*[—\-–:]\s*(.+)$/i);
      if (match) {
        out.push(`${match[1]}. ${match[2].toUpperCase()} — ${match[3].trim()}`);
        autoNum = parseInt(match[1], 10) + 1;
        continue;
      }

      match = line.match(/^(\d+)[\.\):\-]\s+(.+)$/);
      if (match) {
        out.push(`${match[1]}. ${match[2].trim()}`);
        autoNum = parseInt(match[1], 10) + 1;
        continue;
      }

      match = line.match(/^([A-D])\s*[—\-–:]\s*(.+)$/i);
      if (match) {
        out.push(`${autoNum++}. ${match[1].toUpperCase()} — ${match[2].trim()}`);
        continue;
      }

      if (/^[A-D]$/i.test(line)) {
        out.push(`${autoNum++}. ${line.toUpperCase()}`);
        continue;
      }

      out.push(line);
    }

    return out.slice(0, expectedCount || out.length).join('\n');
  }

  formatStandardMcqAnswer(text, expectedCount = null) {
    const lines = this.expandInlineMcqAnswers(text).split('\n').map((line) => line.trim()).filter(Boolean);
    const out = [];
    let autoNum = 1;

    for (const line of lines) {
      let match = line.match(/^(\d+[a-e]?)[\.\):\-]\s+([A-D](?:\s*,\s*[A-D])*)\s*[—\-–:]\s*(.+)$/i);
      if (match) {
        out.push(`${match[1]}. ${match[2].toUpperCase()} — ${match[3].trim()}`);
        continue;
      }

      match = line.match(/^(\d+[a-e]?)[\.\):\-]\s+([A-D])\s*[—\-–:]\s*(.+)$/i);
      if (match) {
        out.push(`${match[1]}. ${match[2].toUpperCase()} — ${match[3].trim()}`);
        autoNum = parseInt(match[1], 10) + 1;
        continue;
      }

      match = line.match(/^(\d+[a-e]?)[\.\):\-]\s+(.+)$/i);
      if (match) {
        const body = match[2].trim();
        const letterOnly = body.match(/^([A-D])$/i);
        if (letterOnly) {
          out.push(`${match[1]}. ${letterOnly[1].toUpperCase()}`);
        } else {
          out.push(`${match[1]}. ${body}`);
        }
        autoNum = parseInt(match[1], 10) + 1;
        continue;
      }

      match = line.match(/^([A-D])\s*[—\-–:]\s*(.+)$/i);
      if (match) {
        out.push(`${autoNum++}. ${match[1].toUpperCase()} — ${match[2].trim()}`);
        continue;
      }

      if (/^[A-D]$/i.test(line)) {
        out.push(`${autoNum++}. ${line.toUpperCase()}`);
        continue;
      }

      out.push(line);
    }

    return out.slice(0, expectedCount || out.length).join('\n');
  }

  stripModelThinking(text) {
    if (!text || typeof text !== 'string') return '';

    let cleaned = text;
    const thinkOpen = '<' + 'think' + '>';
    const thinkEnd = '<' + '/think' + '>';
    const thinkClose = cleaned.lastIndexOf(thinkEnd);
    if (thinkClose !== -1) {
      cleaned = cleaned.slice(thinkClose + thinkEnd.length);
    }

    cleaned = cleaned
      .replace(new RegExp(`${thinkOpen}[\\s\\S]*?${thinkEnd}`, 'gi'), '')
      .replace(new RegExp(`${thinkOpen}[\\s\\S]*`, 'gi'), '')
      .replace(/^\s*<think>[\s\S]*?<\/think>\s*/i, '')
      .trim();

    return cleaned;
  }

  normalizeModelContent(message, requestOptions = null) {
    if (!message) return '';

    let content = message.content;
    if (Array.isArray(content)) {
      content = content
        .filter((part) => part && part.type === 'text' && part.text)
        .map((part) => part.text)
        .join('\n');
    }

    if (typeof content !== 'string') {
      content = '';
    }

    if (!content.trim() && typeof message.reasoning === 'string') {
      content = message.reasoning;
    }

    const stripped = this.stripModelThinking(content);
    if (stripped.trim()) {
      return stripped;
    }

    if (requestOptions && requestOptions.noThinkingSalvage) {
      return '';
    }

    if (content.trim()) {
      const salvaged = this.salvageTextFromThinking(content);
      if (salvaged.trim()) return salvaged;
      return content.trim();
    }

    return '';
  }

  salvageTextFromThinking(text) {
    if (!text || typeof text !== 'string') return '';

    const thinkOpen = '<' + 'think' + '>';
    const thinkEnd = '<' + '/think' + '>';
    const openIdx = text.indexOf(thinkOpen);
    if (openIdx === -1) return text;

    const inner = text.slice(openIdx + thinkOpen.length);
    const closeIdx = inner.indexOf(thinkEnd);
    const body = closeIdx === -1 ? inner : inner.slice(0, closeIdx);

    const lines = body
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => line.length > 0)
      .filter((line) => !/^(let me|wait|okay|hmm|the user|i need|analyz)/i.test(line));

    return lines.join('\n');
  }

  shouldUseVisionReasoningPath(extractedText, rawExtractedText = '', screenshotType = 'other') {
    if (screenshotType === 'text_mcq' || screenshotType === 'math_mcq') {
      return false;
    }

    const text = (extractedText || '').trim();
    const raw = (rawExtractedText || '').trim();
    const blob = `${text}\n${raw}`;

    const isTextualQuiz = /python mcq|multiple[- ]choice|subject-verb|mcq test|mcq quiz|english multiple choice/i.test(blob) &&
      !/grid|matrix|missing piece|shape puzzle|logic puzzle/i.test(blob);

    if (isTextualQuiz) return false;

    if (text.length === 0) {
      return /shape|triangle|square|circle|grid|matrix|pattern|figure|\(\d\)|option\s*\d|\?/i.test(raw);
    }

    if (this.isMultipleChoiceContent(text)) {
      if (/\(\d\)/.test(blob) && /shape|triangle|square|circle|grid|matrix|pattern|figure|\?/i.test(blob)) {
        return true;
      }
    }

    if (text.length < 120 && /shape|triangle|square|circle|grid|matrix|pattern|figure|option\s*\(?\d|\(\d\)|\?/i.test(blob)) {
      return true;
    }

    if (/logic puzzle|visual puzzle|missing piece|3x3|3\s*[x×]\s*3/i.test(blob)) {
      return true;
    }

    return false;
  }

  extractMcqAnswerLetters(text) {
    const letters = [];
    for (const line of (text || '').split('\n')) {
      const match = line.trim().match(/^(?:(\d+)[\.\):\-]\s*)?([A-D])\b/i);
      if (match) {
        letters.push(match[2].toUpperCase());
      }
    }
    return letters;
  }

  isPatternMcqGarbage(text, expectedCount = null, letterOnlyMode = true) {
    if (!text || typeof text !== 'string' || text.trim().length === 0) {
      return true;
    }

    const letters = this.extractMcqAnswerLetters(text);
    if (letters.length === 0) return letterOnlyMode;
    if (letters.length < 3) return false;

    const cycle = ['A', 'B', 'C', 'D'];
    const isPerfectCycle = letters.every((letter, index) => letter === cycle[index % 4]);
    if (isPerfectCycle) return true;

    if (expectedCount && letters.length !== expectedCount) {
      return true;
    }

    return false;
  }

  expandInlineMcqAnswers(text) {
    if (!text || typeof text !== 'string') return '';
    return text
      .replace(/;\s*(?=\d+[a-e]?[\.\):\-]\s+)/gi, '\n')
      .replace(/;\s*(?=Option\s+\d)/gi, '\n');
  }

  countMcqAnswerEntries(text) {
    const expanded = this.expandInlineMcqAnswers(text);
    return this.extractMcqAnswerLines(expanded).length;
  }

  classifyMcqType(text) {
    if (!text) return 'standard';

    const letterOptionLines = (text.match(/(?:^|\n)\s*[a-d][\.\)]\s+/gi) || []).length;
    const numberedQuestions = new Set();
    for (const match of text.matchAll(/(?:^|\n)\s*(\d{1,2})\.\s+/g)) {
      numberedQuestions.add(parseInt(match[1], 10));
    }
    if (letterOptionLines >= 6 || numberedQuestions.size >= 3) {
      return 'standard';
    }

    const reasoningSignals = /(?:shape|triangle|square|circle|grid|matrix|sequence|figure|diagram|visual puzzle|logic puzzle|missing piece|which figure|\(\d\)\s*$|option\s+\d+\s*[—\-–:])/im;
    const mathSignals = /solve\.|×|÷|\$\d|=\s*$|equation|fraction|closest to \d|\d+\s*[x×]\s*\d+/i;

    if (reasoningSignals.test(text)) return 'reasoning';
    if (mathSignals.test(text)) return 'math';
    return 'standard';
  }

  getMcqSystemPrompt(mcqType, questionCount = 1) {
    if (mcqType === 'math') {
      return `You solve multiple choice math questions from screenshots.

OUTPUT:
- Number each answer (1 through ${questionCount})
- Give the ACTUAL answer value AND the option letter: e.g. "1. B — 1225" or "1. $70.05 (B)"
- Do the math for every question — never guess or cycle A,B,C,D mechanically
- No long explanations, no repeating the full question`;
    }

    if (mcqType === 'reasoning') {
      return `You solve visual logic, pattern, and reasoning multiple choice questions from screenshots.

OUTPUT — your ENTIRE reply must be ONE line only:
Option N — [what that option shows] because [one short reason]

Rules:
- Replace N with the correct option number (1, 2, 3, or 4)
- No bullet points, no headers, no step-by-step analysis, no "let me look"
- Maximum 2 short sentences`;
    }

    return `You solve multiple choice questions from screenshots.

OUTPUT:
- Format EVERY answer as: "N. LETTER — answer text" or "Na. LETTER — answer text" for sub-parts (e.g. "1a. A — continue")
- For "select all that apply", list all correct letters: "1c. B, E — dictionary; list"
- Include BOTH the option letter(s) AND the answer phrase for each question
- Answer ALL ${questionCount} question(s) visible in the screenshot
- Never output bare letters only unless the answer text is truly unknown
- Be concise — no repeating the full question or all options`;
  }

  isReasoningAnalysisDump(text) {
    if (!text || typeof text !== 'string') return true;
    const cleaned = text.trim();
    if (cleaned.length > 320) return true;
    if (/^\*\*\d+\.|^-\s+\*\*|^-\s+\*\*Row/m.test(cleaned)) return true;
    if (/(?:let's look|let me look|wait,|analyze the|re-examine|looking at the)/i.test(cleaned)) return true;
    if ((cleaned.match(/^-\s/gm) || []).length >= 2) return true;
    return false;
  }

  formatReasoningMcqAnswer(text) {
    const cleaned = this.stripModelThinking(text).trim();
    if (!cleaned) return '';

    const oneLine = cleaned.split('\n').map((line) => line.trim()).find((line) =>
      /^Option\s*\(?\d\)?\s*[—\-–:]/i.test(line)
    );
    if (oneLine) return oneLine.replace(/^Option\s*\(?(\d)\)?/i, 'Option $1');

    const direct = cleaned.match(/Option\s*\(?(\d)\)?\s*[—\-–:]\s*(.{5,220})/i);
    if (direct) {
      const reason = direct[2].split(/\n/)[0].trim();
      return `Option ${direct[1]} — ${reason}`;
    }

    const conclusion = cleaned.match(
      /(?:answer is|correct (?:option|answer)(?: is)?|(?:so|therefore)[,:]?\s*(?:choose|pick)?)\s*option\s*\(?(\d)\)?[:\s—\-–]*(.*?)(?:\.|$)/i
    );
    if (conclusion) {
      const tail = (conclusion[2] || 'matches the pattern').split('\n')[0].trim();
      return `Option ${conclusion[1]} — ${tail}`;
    }

    const mentions = [...cleaned.matchAll(/option\s*\(?(\d)\)?/gi)];
    if (mentions.length === 1) {
      return `Option ${mentions[0][1]} — matches the row pattern (outer/middle/inner from columns 2/1/2)`;
    }

    return cleaned;
  }

  isMcqResponseValid(text, mcqType, expectedCount = null) {
    const cleaned = this.stripModelThinking(text);
    if (!cleaned.trim()) return false;

    if (mcqType === 'reasoning') {
      if (/<think>|<\/think>/i.test(cleaned)) return false;
      if (this.isReasoningAnalysisDump(cleaned)) return false;

      const formatted = this.formatReasoningMcqAnswer(cleaned);
      if (!/^Option\s*\(?\d\)?\s*[—\-–:]/i.test(formatted)) return false;
      if (formatted.length > 320) return false;
      if (this.isReasoningAnalysisDump(formatted)) return false;
      return true;
    }

    if (mcqType === 'standard') {
      if (this.hasSubstantiveMcqAnswers(cleaned, expectedCount)) {
        const letters = this.extractMcqAnswerLetters(cleaned);
        if (letters.length >= 3 && this.isPatternMcqGarbage(cleaned, expectedCount, true)) {
          return false;
        }
        return true;
      }
      return !this.isPatternMcqGarbage(cleaned, expectedCount, true);
    }

    if (this.isPatternMcqGarbage(cleaned, expectedCount, true)) {
      return false;
    }

    const lines = cleaned.split('\n').map((l) => l.trim()).filter(Boolean);
    if (lines.length === 0) return false;

    if (mcqType === 'math') {
      if (this.hasSubstantiveMcqAnswers(cleaned, expectedCount)) {
        const letters = this.extractMcqAnswerLetters(cleaned);
        if (letters.length >= 3 && this.isPatternMcqGarbage(cleaned, expectedCount, true)) {
          return false;
        }
        return lines.some((line) => /[\d$¼¾½⅓⅔]/.test(line) || /—|-\s*\S/.test(line));
      }
      return false;
    }

    return true;
  }

  buildMcqRetryPayload(extractedText, questionCount, mcqType = 'standard') {
    const typeInstructions = {
      math: `- Compute each answer and output: "N. LETTER — value" (e.g. "1. B — 1225")
- Include the real numeric/monetary/fraction answer, not just the letter`,
      reasoning: `- Identify the correct option number and describe it or explain the pattern in 1-2 sentences
- Example: "Option 1 — large square with triangle inside and small square at center"`,
      standard: `- Give the actual answer text when possible (e.g. "Confidence"), not just "A"
- Format: "N. answer" or "N. LETTER — answer"`
    };

    return `Solve this ${questionCount}-question multiple choice quiz from a screenshot.

RULES:
${typeInstructions[mcqType] || typeInstructions.standard}
- Solve EACH question individually — do not cycle A,B,C,D mechanically
- Do not repeat the full quiz text

Quiz:
${extractedText}`;
  }

  async solveMcqWithEscalation(initialMessages, activeSkill, requestOptions) {
    const questionCount = requestOptions.mcqQuestionCount;
    const mcqType = requestOptions.mcqType || 'standard';
    const extractedText = requestOptions.mcqExtractedText || '';
    const messageSets = [
      initialMessages,
      this.buildGroqRequest(
        this.buildMcqRetryPayload(extractedText, questionCount, mcqType),
        activeSkill,
        [],
        null,
        requestOptions
      )
    ];

    let bestResponse = null;
    let bestScore = 0;

    const modelSequence = ['openai/gpt-oss-120b', 'qwen/qwen3.6-27b'];

    for (let pass = 0; pass < messageSets.length; pass++) {
      const model = modelSequence[Math.min(pass, modelSequence.length - 1)];
      try {
        const response = await this.executeRequest(messageSets[pass], false, activeSkill, {
          ...requestOptions,
          forceModel: model,
          lowTemperature: true
        });
        const raw = this.stripModelThinking(response);
        const cleaned = mcqType === 'standard'
          ? this.formatStandardMcqAnswer(raw, questionCount)
          : mcqType === 'reasoning'
            ? this.formatReasoningMcqAnswer(raw)
            : mcqType === 'math'
              ? this.formatMathMcqAnswer(raw, questionCount)
              : raw.trim();

        const score = this.scoreMcqResponse(cleaned, mcqType, questionCount);
        if (score > bestScore) {
          bestScore = score;
          bestResponse = cleaned;
        }

        if (this.isMcqResponseValid(cleaned, mcqType, questionCount)) {
          logger.info('MCQ solved successfully', { model, mcqType, pass: pass + 1, questionCount, answers: cleaned });
          return cleaned;
        }

        logger.warn('MCQ model returned invalid response', {
          model,
          mcqType,
          pass: pass + 1,
          response: cleaned
        });
      } catch (error) {
        logger.warn('MCQ model attempt failed', {
          model,
          mcqType,
          pass: pass + 1,
          error: error.message
        });
      }
    }

    if (bestResponse && bestScore >= 15) {
      logger.warn('MCQ returning best partial response', { bestScore, questionCount, mcqType, answers: bestResponse });
      return bestResponse;
    }

    throw new Error('Could not solve quiz reliably. Please try the screenshot again.');
  }

  async answerReasoningMcqFromImage(imageSources, activeSkill, extractedText = '') {
    let patternContext = '';
    if (extractedText && extractedText.trim().length > 0) {
      const lines = extractedText.split('\n').filter((line) =>
        /outer|inner|column|row|pattern|option\s*\(?\d\)?/i.test(line)
      );
      if (lines.length > 0) {
        patternContext = `\n\nExtracted puzzle notes (use as hints, verify visually):\n${lines.slice(0, 12).join('\n')}`;
      }
    }

    const instruction = {
      type: 'text',
      text: `Look at this visual logic / pattern puzzle in the image.

Find the correct answer among the numbered options (1, 2, 3, 4) below the grid.

For each row, column 3 is usually a triple-nested shape:
- OUTER = column 2 shape
- MIDDLE = column 1 shape  
- INNER = column 2 shape again

Check all four options carefully — pick the one whose outer, middle, AND inner shapes all match.

Reply with EXACTLY one line (nothing else):
Option N — [outer > middle > inner shapes] because [one sentence why it fits the row pattern]${patternContext}`
    };

    const content = [instruction];

    if (typeof imageSources === 'string') {
      content.push({ type: 'image_url', image_url: { url: imageSources } });
    } else if (Array.isArray(imageSources)) {
      imageSources.forEach((img) => {
        if (typeof img === 'string') {
          content.push({ type: 'image_url', image_url: { url: img } });
          return;
        }
        const base64Image = img.imageBuffer.toString('base64');
        const url = `data:${img.mimeType || 'image/png'};base64,${base64Image}`;
        content.push({ type: 'image_url', image_url: { url } });
      });
    }

    const messages = [
      { role: 'system', content: this.getMcqSystemPrompt('reasoning', 1) },
      { role: 'user', content }
    ];

    const models = [this.visionModel];
    for (const model of models) {
      try {
        const response = await this.executeRequest(messages, true, activeSkill, {
          lowTemperature: true,
          forceModel: model,
          mcqType: 'reasoning',
          maxTokensOverride: 450,
          noThinkingSalvage: true
        });
        const formatted = this.formatReasoningMcqAnswer(response);
        if (this.isMcqResponseValid(formatted, 'reasoning', 1)) {
          logger.info('Visual reasoning solved via vision', { model, response: formatted });
          return formatted;
        }
        logger.warn('Vision reasoning response invalid', { model, preview: formatted.substring(0, 120) });
      } catch (error) {
        logger.warn('Vision reasoning model failed', { model, error: error.message });
      }
    }

    throw new Error('Could not solve visual puzzle from image.');
  }

  async tryReasoningMcqVisionPath(extractedText, imageSource, activeSkill, startTime, programmingLanguage, mimeType = null, rawExtractedText = '', screenshotType = 'other') {
    if (!this.shouldUseVisionReasoningPath(extractedText, rawExtractedText, screenshotType)) {
      return null;
    }

    logger.info('[Pipeline Stage 2] Solving visual reasoning via Qwen Vision', {
      extractedTextLength: (extractedText || '').length,
      rawExtractedLength: (rawExtractedText || '').length
    });
    const reasoningResponse = await this.answerReasoningMcqFromImage(imageSource, activeSkill, extractedText);

    return {
      response: reasoningResponse,
      metadata: {
        skill: activeSkill,
        programmingLanguage,
        processingTime: Date.now() - startTime,
        requestId: this.requestCount,
        usedFallback: false,
        isImageAnalysis: true,
        isMcq: true,
        mcqType: 'reasoning',
        mimeType
      }
    };
  }

  enforceMcqAnswerOnly(text, expectedCount = null) {
    if (!text || typeof text !== 'string') return '';

    const lines = text.split('\n').map((line) => line.trim()).filter(Boolean);
    const answers = [];

    for (const line of lines) {
      const match = line.match(/^(?:(\d+)[\.\):\-]\s*)?([ABCD])\b/i);
      if (match) {
        const num = match[1] ? parseInt(match[1], 10) : answers.length + 1;
        answers.push({ num, letter: match[2].toUpperCase() });
      }
    }

    if (answers.length === 0) {
      const soloLetters = text.match(/\b([ABCD])\b/gi);
      if (soloLetters && soloLetters.length > 0) {
        const limit = expectedCount || soloLetters.length;
        return soloLetters
          .slice(0, limit)
          .map((letter, index) => `${index + 1}. ${letter.toUpperCase()}`)
          .join('\n');
      }
      return text;
    }

    answers.sort((a, b) => a.num - b.num);
    const uniqueByNum = [];
    const seenNums = new Set();
    for (const answer of answers) {
      if (seenNums.has(answer.num)) continue;
      seenNums.add(answer.num);
      uniqueByNum.push(answer);
    }

    const limit = expectedCount || uniqueByNum.length;
    return uniqueByNum
      .slice(0, limit)
      .map((answer) => `${answer.num}. ${answer.letter}`)
      .join('\n');
  }

  getMcqImageSystemPrompt(mcqType = 'standard', questionCount = 1) {
    return this.getMcqSystemPrompt(mcqType, questionCount);
  }

  shouldTryNextApiKey(errorInfo) {
    return [
      'RATE_LIMIT_ERROR',
      'REQUEST_TOO_LARGE_ERROR',
      'AUTH_ERROR',
      'NETWORK_ERROR',
      'SERVER_ERROR',
      'UNKNOWN_ERROR'
    ].includes(errorInfo.type);
  }

  buildGeneralImageGenerationPayload(extractedText, options = {}) {
    const screenshotType = options.screenshotType || 'other';
    const isCodingExercise = this.isCodingExerciseContent(extractedText);
    const isMcq = this.isMultipleChoiceContent(extractedText) ||
      screenshotType === 'text_mcq' ||
      screenshotType === 'math_mcq';
    const detectedLang = this.detectProgrammingLanguageFromText(extractedText);
    const langNote = detectedLang
      ? ` Use ${detectedLang.toUpperCase()} only.`
      : '';

    if (isCodingExercise) {
      return {
        payload: `This is a coding exercise extracted from a screenshot. Write the COMPLETE working solution.${langNote}

RULES:
- Output ONLY the final code inside a single code block
- Do NOT repeat requirements, UI text, URLs, or the extracted content
- Do NOT explain anything
- Include everything needed to satisfy frontend and backend requirements when both are present

Reference requirements (do not repeat these in your answer):
${extractedText}`,
        detectedLang,
        isCodingExercise: true,
        isMcq: false,
        questionCount: 0
      };
    }

    if (isMcq) {
      const questionCount = this.countMcqQuestions(extractedText);
      const mcqType = screenshotType === 'math_mcq'
        ? 'math'
        : screenshotType === 'text_mcq'
          ? 'standard'
          : this.classifyMcqType(extractedText);
      const payloadByType = {
        math: `MULTIPLE CHOICE MATH QUIZ — ${questionCount} question(s).

Solve each problem. Output numbered answers with the REAL value and option letter.
Format: "1. B — 198÷2" or "1. C — 1225"
Do NOT output bare letters only. Do NOT cycle A,B,C,D mechanically.

Quiz:
${extractedText}`,
        reasoning: `VISUAL / LOGIC MULTIPLE CHOICE — solve from the screenshot content.

Identify the correct option and explain briefly based on the pattern you see.
Format: "Option 1 — [what it looks like or why]" or "1. Option 1 — brief reason"
Be concise. The answer should make sense without seeing the image.

Content:
${extractedText}`,
        standard: `MULTIPLE CHOICE QUIZ — ${questionCount} question(s).

Give the correct option letter AND answer text for every visible question.
Format: "1. B — Id returns the identity of the object" or "1a. A — continue" for sub-parts.
For "select all that apply", use: "1c. B, E — dictionary; list"
Do NOT output bare letters only. Answer all ${questionCount} questions.
${/python/i.test(extractedText) ? '\nPython note: keywords include True, False, None (capitalized) — not all lowercase and not all UPPERCASE.' : ''}

Quiz:
${extractedText}`
      };

      return {
        payload: payloadByType[mcqType] || payloadByType.standard,
        detectedLang: null,
        isCodingExercise: false,
        isMcq: true,
        mcqType,
        questionCount,
        extractedText
      };
    }

    return {
      payload: `Answer the following content from a screenshot. Output ONLY the final answer.

RULES:
- Multiple choice: output ONLY the correct letter (A/B/C/D)
- Math/problem: output ONLY the final result
- Do NOT repeat extracted text, UI chrome, URLs, or copyright notices
- Do NOT explain

Content:
${extractedText}`,
      detectedLang: null,
      isCodingExercise: false,
      isMcq: false,
      questionCount: 0
    };
  }

  async processTextWithSkill(text, activeSkill, sessionMemory = [], programmingLanguage = null, requestOptions = null) {
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

      const messages = this.buildGroqRequest(text, activeSkill, sessionMemory, programmingLanguage, requestOptions);

      let finalResponse;
      if (requestOptions && requestOptions.mcqAnswerOnly) {
        finalResponse = await this.solveMcqWithEscalation(messages, activeSkill, requestOptions);
      } else {
        const responseText = await this.executeRequestWithFallback(messages, false, activeSkill, requestOptions);
        finalResponse = programmingLanguage
          ? this.enforceProgrammingLanguage(responseText, programmingLanguage)
          : responseText;
      }

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

      if (config.get('llm.groq.fallbackEnabled') && !(requestOptions && requestOptions.mcqAnswerOnly)) {
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
      const responseText = await this.executeRequestWithFallback(messages, false, activeSkill);

      const codingSkills = ['dsa', 'programming'];
      const humanizedResponse = !codingSkills.includes(activeSkill)
        ? this.applyHumanizerFixes(responseText)
        : responseText;

      const finalResponse = programmingLanguage
        ? this.enforceProgrammingLanguage(humanizedResponse, programmingLanguage)
        : humanizedResponse;

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

  buildGroqRequest(text, activeSkill, sessionMemory, programmingLanguage, requestOptions = null) {
    const sessionManager = require('../managers/session.manager');

    const codingSkills = ['dsa', 'programming'];

    if (sessionManager && typeof sessionManager.getConversationHistory === 'function') {
      // For coding skills, skip history entirely — each question is self-contained
      // and past code responses are too large and push requests over the TPM limit.
      const conversationHistory = codingSkills.includes(activeSkill) || (requestOptions && requestOptions.mcqAnswerOnly)
        ? []
        : sessionManager.getConversationHistory(4);
      const skillContext = sessionManager.getSkillContext(activeSkill, programmingLanguage);
      return this.buildGroqRequestWithHistory(text, activeSkill, conversationHistory, skillContext, programmingLanguage, requestOptions);
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

  buildGroqRequestWithHistory(text, activeSkill, conversationHistory, skillContext, programmingLanguage, requestOptions = null) {
    const messages = [];
    const sessionManager = require('../managers/session.manager');
    const documentContext = sessionManager.getDocumentContext();

    if (documentContext) {
      messages.push({ role: 'system', content: `## Reference Document Context\n${documentContext}\n\n## FIRST-PERSON RULE\nYou must adopt a first-person persona based on the reference document context provided above. When answering questions, speak directly from the perspective of the document's subject or author. Use "I", "me", "my". Keep your answers extremely concise (not too long, not too short). Do not break character.` });
    }

    const isImageDerivedAnswer = /this is a coding exercise extracted from a screenshot|answer the following content from a screenshot|MULTIPLE CHOICE QUIZ/i.test(text);
    const isCodingContent = this.isCodingExerciseContent(text);
    const isMcqContent = (requestOptions && requestOptions.mcqAnswerOnly) || this.isMultipleChoiceContent(text);
    const isBasicGeneralQuestion = activeSkill === 'general' && !isImageDerivedAnswer && !isCodingContent && !isMcqContent;

    if (isMcqContent && activeSkill === 'general') {
      const mcqType = (requestOptions && requestOptions.mcqType) || this.classifyMcqType(text);
      const questionCount = (requestOptions && requestOptions.mcqQuestionCount) || this.countMcqQuestions(text);
      messages.push({
        role: 'system',
        content: this.getMcqSystemPrompt(mcqType, questionCount)
      });
    } else if (isBasicGeneralQuestion) {
      messages.push({
        role: 'system',
        content: this.getHumanizedBasicQuestionPrompt(sessionManager.getResponseMode(), documentContext)
      });
    } else if (skillContext.skillPrompt) {
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

  /**
   * Port of copy_scan.py (humanizer-stack) to JavaScript.
   * Applies deterministic, zero-latency surface fixes to any LLM response.
   * No second LLM call. Pure string operations.
   */
  applyHumanizerFixes(text) {
    if (!text || typeof text !== 'string') return text;

    let result = text;

    // 1. Em dash fix — replace " — " or "word—word" with ", " or ", "
    // Pattern from copy_scan.py: copy-em-dash  (\w\s*—\s*\w)
    result = result.replace(/(\w)\s*—\s*(\w)/g, (_, before, after) => `${before}, ${after}`);
    result = result.replace(/(\w)\s*—\s*/g, (_, before) => `${before}, `);
    result = result.replace(/\s*—\s*(\w)/g, (_, after) => `, ${after}`);
    result = result.replace(/—/g, ','); // catch any remaining em dashes

    // 2. Antithesis / negative parallelism fix — copy-antithesis
    // "it's not just X, it's Y" → strip the negation preamble, keep "it's Y"
    result = result.replace(
      /(?:it'?s |it |that'?s )?not just[^.,;!?]{1,40},?\s*(it'?s|but)\s*/gi,
      '$1 '
    );
    result = result.replace(/\bnot only\b([^.,;!?]{1,40}),?\s*but\b/gi, '$1 and');

    // 3. Hype / marketing cliche words — from copy_scan.py hype-copy
    const hypePhrases = [
      [/\bTransform your\b/gi, 'Improve your'],
      [/\bSupercharge\b/gi, 'Improve'],
      [/\bUnleash\b/gi, 'Use'],
      [/\bEffortlessly\b/gi, 'Easily'],
      [/\breimagined\b/gi, 'redesigned'],
      [/\bGame-?changer\b/gi, 'big improvement'],
      [/\bdelve\b/gi, 'get into'],
      [/\bdive into the details\b/gi, 'get into it'],
      [/\bdive into\b/gi, 'get into'],
      [/\belevate your\b/gi, 'improve your'],
      [/\bworld-class\b/gi, 'top-tier'],
      [/\bcutting-edge\b/gi, 'modern'],
      [/\brevolutionary\b/gi, 'new'],
      [/\bbest-in-class\b/gi, 'top'],
      // From humanizer SKILL.md banned word list
      [/\bthrive\b/gi, 'do well'],
      [/\bpivotal\b/gi, 'important'],
      [/\brobust\b/gi, 'solid'],
      [/\bleverage\b/gi, 'use'],
      [/\bseamless\b/gi, 'smooth'],
      [/\bmultifaceted\b/gi, 'complex'],
    ];
    for (const [pat, replacement] of hypePhrases) {
      result = result.replace(pat, replacement);
    }

    // 4. Servile openers — copy-servile
    result = result.replace(/^(Great question[!.]*\s*)/i, '');
    result = result.replace(/^(I hope this helps[!.]*\s*)/i, '');
    result = result.replace(/^(Certainly[!,]*\s*)/i, '');
    result = result.replace(/^(Of course[!,]*\s*)/i, '');
    result = result.replace(/^(Sure thing[!,]*\s*)/i, '');
    result = result.replace(/^(Happy to (?:help|assist|answer|explain)[!,]*\s*)/i, '');
    result = result.replace(/^(Absolutely[!,]*\s*)/i, '');
    result = result.replace(/^(Great[!,]\s*)/i, '');
    // Inline servile phrases at the start of answers
    result = result.replace(/^((?:Sure thing|Happy to)[^.!?]{0,40},\s*)/i, '');

    // 5. Tidy closer patterns — from structural_scan.py TIDY_CLOSER + observed failures
    // These appear as the LAST sentence. Remove them.
    const tidyCloserPats = [
      /[.!]?\s*(?:Ultimately|In the end|At the end of the day)[^.!?]*[.!?]\s*$/i,
      /[.!]?\s*(?:The skill sets? overlap[^.!?]*)[.!?]\s*$/i,
      /[.!]?\s*(?:The focus was on[^.!?]*)[.!?]\s*$/i,
      /[.!]?\s*(?:Overall[,]? (?:my|our|the)[^.!?]*)[.!?]\s*$/i,
      /[.!]?\s*(?:That(?:'s| is) the main difference)[.!?]\s*$/i,
      /[.!]?\s*(?:That(?:'s| is) what (?:I|we) (?:always?|really) (?:keep|kept)[^.!?]*)[.!?]\s*$/i,
      // Observed failures batch 1:
      /[.!]?\s*It was a mix of[^.!?]*[.!?]\s*$/i,
      /[.!]?\s*So while (?:my|our|the)[^.!?]*[.!?]\s*$/i,
      /[.!]?\s*I'm (?:ready|excited|happy) to (?:dive|jump|start)[^.!?]*[.!?]\s*$/i,
      /[.!]?\s*That'?s? (?:the|my) main (?:focus|goal|priority|approach)[^.!?]*[.!?]\s*$/i,
      /[.!]?\s*(?:That|This) (?:contrast|difference|distinction) always[^.!?]*[.!?]\s*$/i,
      /[.!]?\s*I thrive on[^.!?]*[.!?]\s*$/i,
      // Observed failures batch 2:
      /[.!]?\s*(?:So )?the focus shifts from[^.!?]*[.!?]\s*$/i,
      /[.!]?\s*I'm ready to start (?:immediately|right away)[^.!?]*[.!?]\s*$/i,
      /[.!]?\s*(?:I|we) do well in[^.!?]*[.!?]\s*$/i,
      /[.!]?\s*keeping the (?:design|process|work|project)[^.!?]*(?:track|path|schedule)[^.!?]*[.!?]\s*$/i,
    ];
    for (const pat of tidyCloserPats) {
      result = result.replace(pat, '');
    }

    // 6. Oxford comma Rule of Three reducer — trim "X, Y, and Z" to "X and Y"
    // Only applies to short noun/verb phrase lists (under 60 chars total for safety)
    // Pattern: word/phrase COMMA word/phrase COMMA "and" word/phrase
    result = result.replace(
      /([^,\n]{3,30}),\s+([^,\n]{3,30}),\s+and\s+([^,.!?\n]{3,30})/g,
      (match, a, b, c) => {
        // Only trim if all three segments are short (likely a flat list, not a complex clause)
        if (a.length < 35 && b.length < 35 && c.length < 35) {
          return `${a} and ${b}`;
        }
        return match; // leave complex sentences alone
      }
    );

    return result.trim();
  }

  getHumanizedBasicQuestionPrompt(mode, documentContext = null) {
    let prompt = `You are a real person in a live interview conversation. Speak naturally and directly.

HOW TO SPEAK:
- Use contractions naturally: I'm, it's, don't, I've.
- React to what you're saying — don't just report facts. Occasionally add a personal note: "honestly that part was tricky", "I didn't expect that to matter as much", "that's what kept me there".
- Don't end answers with a summary of yourself or a conclusion sentence. Just stop when the last fact is stated.
- Use specific names, numbers, and products. Name the actual software, the actual metric.
- Never open with "Certainly!", "Great question!", or "I'd be happy to help".
- Don't perform emotion through the body. Say the feeling plainly.

FORMAT RULES:`;

    if (mode === 'simple') {
      prompt += `\n- Answer in 1-2 sentences only.\n- No bullet points, numbered lists, headers, bold, markdown, or code.\n- Plain flowing spoken sentences only.`;
    } else if (mode === 'medium') {
      prompt += `\n- Answer in 1-2 short paragraphs.\n- No bullet points, numbered lists, headers, bold, markdown, or code.\n- Plain flowing spoken sentences only.`;
    } else {
      prompt += `\n- Answer in 2-3 short paragraphs max. Each paragraph is 2 sentences max.\n- Blend ideas together — don't give every sub-topic its own paragraph.\n- No bullet points, numbered lists, headers, bold, markdown, or code.\n- For simple questions (yes/no, naming, listing), keep it to 1-2 sentences.\n- Plain flowing spoken sentences only.`;
    }

    if (documentContext) {
      prompt += `\n\nStay in character as the person described below. Use "I", "me", "my". Do not break character.\n\n${documentContext}`;
    }

    return prompt;
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

    const humanizedPrompt = this.getHumanizedBasicQuestionPrompt(mode, documentContext);

    // Extract the candidate's name from the document context for an explicit identity anchor
    let nameInstruction = 'IDENTITY: You are the candidate in this interview. You do NOT have a name like "ChatGPT" or "AI". If asked your name and no reference document is loaded, deflect naturally (e.g. "You can just call me by my first name — what else can I help with?").';
    if (documentContext) {
      const nameMatch = documentContext.match(/(?:name[:\s]+|^)([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)/m);
      if (nameMatch) {
        nameInstruction = `IDENTITY: Your name is ${nameMatch[1]}. You are ${nameMatch[1]}, the candidate in this interview. Always use this name when asked who you are.`;
      } else {
        nameInstruction = 'IDENTITY: Stay in character as the candidate described in the reference document below. Use their name if asked.';
      }
    }

    return `You are in a live mock interview. You are the interviewee. The provided text is the real-time transcription of ONLY the interviewer's voice. Follow these rules strictly:
1. Act naturally and stay in character at all times.
2. Respond conversationally to the interviewer's statements and questions.
3. If the interviewer uses a conversational filler (like "Okay") or says goodbye, provide a natural, brief acknowledgment.
4. NEVER output your internal thoughts, reasoning, or justification. Output ONLY your spoken dialogue.
5. Keep your responses concise and do not speak on behalf of the interviewer.
6. ${nameInstruction}

${humanizedPrompt}`;
  }

  formatUserMessage(text, activeSkill) {
    if (
      activeSkill === 'general' &&
      (/this is a coding exercise extracted from a screenshot|answer the following content from a screenshot/i.test(text) ||
        this.isCodingExerciseContent(text))
    ) {
      return text;
    }

    if (activeSkill === 'general') {
      return text;
    }

    return `Context: ${activeSkill.toUpperCase()} analysis request\n\nText to analyze:\n${text}`;
  }

  async executeRequest(messages, isVision = false, activeSkillOverride = null, requestOptions = null) {
    const sessionManager = require('../managers/session.manager');
    const mode = sessionManager.getResponseMode();
    let maxTokens = 2500;
    if (mode === 'simple') maxTokens = 450;
    if (mode === 'medium') maxTokens = 850;

    const activeSkill = activeSkillOverride || sessionManager.currentSkill;
    const isCodingSkill = ['dsa', 'programming'].includes(activeSkill);
    const isGeneralSkill = activeSkill === 'general';
    const preferCodingTokenBudget = Boolean(requestOptions && requestOptions.preferCodingTokenBudget);
    const mcqAnswerOnly = Boolean(requestOptions && requestOptions.mcqAnswerOnly);
    const forceModel = requestOptions && requestOptions.forceModel;

    // Fast model rotation pool — each model has independent rate limits on Groq free tier
    let modelPool = isVision
      ? [this.visionModel]
      : ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'llama3-8b-8192', 'gemma2-9b-it'];

    if (forceModel) {
      modelPool = [forceModel];
    } else if (isCodingSkill) {
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
    } else if (isGeneralSkill) {
      if (isVision) {
        modelPool = [this.visionModel];
        if (mode === 'complex') maxTokens = 2500;
        else if (mode === 'medium') maxTokens = 1500;
        if (requestOptions && requestOptions.mcqType === 'reasoning') {
          maxTokens = 450;
        }
      } else {
        // MCQ: prefer Qwen first — GPT OSS can burn tokens on hidden reasoning and return empty
        modelPool = mcqAnswerOnly
          ? ['qwen/qwen3.6-27b', 'openai/gpt-oss-120b', 'openai/gpt-oss-20b']
          : ['openai/gpt-oss-20b', 'qwen/qwen3.6-27b'];
        if (preferCodingTokenBudget) {
          if (mode === 'complex') maxTokens = 6000;
          else if (mode === 'medium') maxTokens = 4000;
          else maxTokens = 2000;
        } else if (mcqAnswerOnly) {
          if (mode === 'complex') maxTokens = 4000;
          else if (mode === 'medium') maxTokens = 2500;
          else maxTokens = 1200;
          if (requestOptions && requestOptions.mcqType === 'reasoning') {
            maxTokens = Math.min(maxTokens + 500, 4500);
          }
        } else {
          if (mode === 'complex') maxTokens = 2500;
          else if (mode === 'medium') maxTokens = 1200;
          else maxTokens = 600;
        }
      }
    }

    if (requestOptions && Number.isFinite(requestOptions.maxTokensOverride)) {
      maxTokens = requestOptions.maxTokensOverride;
    }

    const payload = {
      messages,
      model: modelPool[0],
      ...this.getGenerationConfig(
        requestOptions && (requestOptions.mcqDeterministic || requestOptions.lowTemperature)
          ? { temperature: 0, top_p: 1 }
          : isGeneralSkill && !isVision && !preferCodingTokenBudget && !mcqAnswerOnly
            ? { temperature: 0.6 }
            : {}
      ),
      max_tokens: maxTokens
    };

    // Qwen 3.6 defaults to thinking mode — burns max_tokens before any answer (vision/MCQ fail with finish_reason=length)
    if (payload.model && payload.model.includes('qwen/qwen3.6-27b') && !(requestOptions && requestOptions.allowQwenThinking)) {
      payload.reasoning_effort = 'none';
    }

    let lastError = null;

    // Round-robin across all 7 keys — never pin to key 0 (that caused rate limits)
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

          const choice = response.choices[0];
          let content = this.normalizeModelContent(choice.message, requestOptions);
          const finishReason = choice.finish_reason;

          if (!content || (typeof content === 'string' && content.trim().length === 0)) {
            throw new Error(`Empty content from Groq API (model=${payload.model}, finish_reason=${finishReason || 'unknown'})`);
          }

          return content;
        } catch (error) {
          lastError = error;
          const errorInfo = this.analyzeError(error);

          logger.warn(`Groq model ${payload.model} failed on API key index ${clientIndex}`, {
            error: error.message,
            errorType: errorInfo.type,
            model: payload.model,
            keyIndex: clientIndex,
            totalKeys: this.clients.length
          });

          if (this.shouldTryNextApiKey(errorInfo) && j < this.clients.length - 1) {
            continue;
          }

          break;
        }
      }

      if (lastError) {
        const errorInfo = this.analyzeError(lastError);

        if (this.shouldTryNextApiKey(errorInfo) && i < modelPool.length - 1) {
          logger.info(`All ${this.clients.length} API keys exhausted for ${payload.model} (${errorInfo.type}), switching to ${modelPool[i + 1]}`);
          continue;
        }

        if (i === modelPool.length - 1) {
          throw new Error(`All Groq models and ${this.clients.length} API keys exhausted: ${lastError.message}`);
        }

        if (errorInfo.type !== 'RATE_LIMIT_ERROR') {
          const delay = 1000 + Math.random() * 500;
          await this.delay(delay);
        }
      }
    }

    throw new Error(`All Groq models and ${this.clients.length} API keys exhausted. Last error: ${lastError ? lastError.message : 'Unknown'}`);
  }

  // 413 fallback — if ALL preferred coding models rejected the request because it was too
  // large, retry once with the standard model pool which has a much higher context limit.
  async executeRequestWithFallback(messages, isVision = false, activeSkillOverride = null, requestOptions = null) {
    const sessionManager = require('../managers/session.manager');
    const activeSkill = activeSkillOverride || sessionManager.currentSkill;
    const isCodingSkill = ['dsa', 'programming'].includes(activeSkill) && !isVision;

    try {
      return await this.executeRequest(messages, isVision, activeSkill, requestOptions);
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
    const status = error.status || error.statusCode;

    if (errorMessage.includes('fetch failed') || errorMessage.includes('network error') || errorMessage.includes('timeout') || errorMessage.includes('econnreset') || errorMessage.includes('enotfound')) {
      return { type: 'NETWORK_ERROR', isNetworkError: true };
    }

    if (status === 401 || status === 403 || errorMessage.includes('unauthorized') || errorMessage.includes('invalid api key') || errorMessage.includes('permission denied')) {
      return { type: 'AUTH_ERROR', isNetworkError: false };
    }

    if (status === 429 || errorMessage.includes('rate limit') || errorMessage.includes('too many requests')) {
      return { type: 'RATE_LIMIT_ERROR', isNetworkError: false };
    }

    if (status === 413 || errorMessage.includes('request too large') || errorMessage.includes('reduce your message size')) {
      return { type: 'REQUEST_TOO_LARGE_ERROR', isNetworkError: false };
    }

    if (status >= 500 || errorMessage.includes('server error') || errorMessage.includes('service unavailable') || errorMessage.includes('bad gateway')) {
      return { type: 'SERVER_ERROR', isNetworkError: true };
    }

    if (errorMessage.includes('decommissioned') || errorMessage.includes('model_not_found') || errorMessage.includes('does not exist') || errorMessage.includes('model not found')) {
      return { type: 'MODEL_ERROR', isNetworkError: false };
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
      apiKeyCount: this.clients.length,
      config: config.get('llm.groq')
    };
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

module.exports = new LLMService();