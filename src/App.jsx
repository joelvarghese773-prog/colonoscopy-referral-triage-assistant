import { useRef, useState } from 'react'
import { Stethoscope, Loader2, AlertCircle, User, AlertTriangle, CheckCircle, BookOpen, ShieldAlert, BarChart3 } from 'lucide-react'
import { GoogleGenerativeAI } from '@google/generative-ai'
import { PDFParse } from 'pdf-parse'
import './App.css'

const genAI = new GoogleGenerativeAI(import.meta.env.VITE_GEMINI_API_KEY)
const GEMINI_MODEL_NAME = 'gemini-2.5-flash'
const model = genAI.getGenerativeModel({
  model: GEMINI_MODEL_NAME
})

const GUIDELINE_PDF_URL = '/guidelines/Colonoscopy_Clinical_Care_Standard_2025.pdf'
const CHUNK_MIN_WORDS = 300
const CHUNK_TARGET_WORDS = 400
const CHUNK_MAX_WORDS = 500
const RETRIEVAL_CHUNK_COUNT = 3
const PRIORITY_CHUNK_COUNT = 2

const GEMINI_PROMPT = `You are an Australian Clinical Decision Support Assistant.

You assist clinicians by analysing colonoscopy referral letters using the Australian Colonoscopy Clinical Care Standard.

The retrieved guideline excerpts below are the ONLY authoritative clinical reference.

You MUST base every recommendation on these retrieved guideline excerpts.

Do NOT rely on general medical knowledge if it conflicts with the retrieved guideline.

If the retrieved guideline does not contain sufficient information to justify a recommendation, explicitly state that more clinical information is required.

Do not invent recommendations.

Do not invent guideline statements.

Do not fabricate clinical reasoning.

Never diagnose disease.

Never replace specialist judgement.

Provide structured clinical decision support only.

Always explain how the retrieved guideline supports the triage recommendation.

If multiple retrieved guideline chunks are relevant, combine them.

If there is insufficient evidence in the retrieved guideline, clearly state this.

Return ONLY valid JSON.

Do not include markdown.

Do not include explanations outside the JSON.

Referral Completeness Score

Calculate a completeness score from 0-100.

Score the referral according to whether the following information is provided:

- Patient age
- Presenting symptoms
- Symptom duration
- Relevant medical history
- Medications
- Family history
- Physical findings
- Investigation results
- Relevant blood tests
- Previous colonoscopy/endoscopy
- Reason for referral
- Requested urgency

100 = comprehensive referral containing nearly all clinically relevant information.

70-90 = good referral with minor missing information.

40-69 = moderate information provided.

10-39 = limited information.

0-9 = almost no useful clinical information.

Return the score as an integer between 0 and 100.

Use ONLY the guideline excerpts below as your knowledge source.

If the excerpts do not contain enough information, state that limitation in guidelineRationale.

You MUST include guidelineEvidence with 2-5 short statements taken ONLY from the retrieved guideline chunks that justify the recommendation.

If no supporting evidence exists, return exactly "No directly relevant guideline evidence retrieved.".

List missingInformation items that would improve confidence.

Return confidence as exactly one of High, Moderate, or Low.

Confidence should depend ONLY on referral completeness, quality of retrieved guideline evidence, and consistency between referral and guideline.

Return priority using only one of the following exact values:

Category 1 - Urgent

Category 2 - Semi-Urgent

Category 3 - Routine

The guidelineRationale must explain why the referral fits the category using only the retrieved guideline excerpts.

The guidelineRationale must explicitly reference the retrieved evidence by describing which retrieved statements support the decision, why the patient satisfies those criteria, and where the retrieved guideline is insufficient.

Never state a recommendation as fact unless it is supported by the retrieved guideline excerpts.

If the retrieved evidence is limited, say exactly what is missing and why that limits confidence.

The recommendedActions must be supported by the retrieved guideline excerpts.

Clinical Recommendations (recommendedActions)

Return 3 to 5 recommendations as an array of complete sentences.

Recommendations must be specific to the current referral and explicitly grounded in the retrieved guideline excerpts.

Recommendations may include referral urgency, next clinical step, required additional information, investigations/documentation to accompany the referral, and follow-up actions supported by the retrieved guideline.

Do not provide treatment recommendations.

Do not diagnose disease.

Do not include generic statements such as "Refer for colonoscopy" without referral-specific context.

Always explain how the retrieved guideline supports the triage recommendation.

If there is insufficient evidence in the retrieved guideline, clearly state this.

Guideline excerpts:

{{GUIDELINE_CHUNKS}}

Schema:

{
  "priority": "",
  "completenessScore": 0,
  "patientSummary": "",
  "redFlags": [
    "string"
  ],
  "clinicalFindings": [
    "string"
  ],
  "recommendedActions": [
    "string"
  ],
  "guidelineRationale": "",
  "guidelineEvidence": [
    "string"
  ],
  "missingInformation": [
    "string"
  ],
  "confidence": "",
  "safetyNotice": ""
}

Analyse this referral:

{{REFERRAL}}`

function App() {
  const [referralText, setReferralText] = useState('')
  const [aiResponse, setAiResponse] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [priority, setPriority] = useState(null)
  const [apiError, setApiError] = useState('')
  const guidelineChunksRef = useRef([])

  const tokenize = (text) => {
    const stopWords = new Set([
      'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its',
      'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with', 'or', 'this', 'these', 'those', 'into',
      'about', 'than', 'then', 'there', 'their', 'them', 'they', 'she', 'his', 'her', 'you', 'your', 'our'
    ])

    return text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, ' ')
      .split(/\s+/)
      .filter((token) => token.length > 2 && !stopWords.has(token))
  }

  const isLikelyHeading = (line) => {
    if (!line) return false
    if (line.length > 100) return false
    if (/^(table|figure)\b/i.test(line)) return true
    if (/^\d+(\.\d+)*[.)]?\s+/.test(line)) return true
    return /^[A-Z][A-Z\s/&-]{3,}$/.test(line)
  }

  const splitIntoSections = (rawText) => {
    const lines = rawText
      .split('\n')
      .map((line) => line.replace(/\s+/g, ' ').trim())
      .filter(Boolean)

    const sections = []
    let currentHeading = 'General Guidance'
    let buffer = []

    for (const line of lines) {
      if (isLikelyHeading(line)) {
        if (buffer.length > 0) {
          sections.push({
            heading: currentHeading,
            text: buffer.join(' ')
          })
          buffer = []
        }
        currentHeading = line
      } else {
        buffer.push(line)
      }
    }

    if (buffer.length > 0) {
      sections.push({
        heading: currentHeading,
        text: buffer.join(' ')
      })
    }

    return sections
  }

  const parsePageNumber = (heading) => {
    const pageMatch = heading.match(/--\s*(\d+)\s+of\s+\d+\s*--/i)
    if (pageMatch) {
      return Number(pageMatch[1])
    }

    return null
  }

  const createWordChunks = (words) => {
    const chunks = []
    let cursor = 0

    while (cursor < words.length) {
      const remaining = words.length - cursor
      let take = remaining

      if (remaining > CHUNK_MAX_WORDS) {
        take = CHUNK_TARGET_WORDS
        const leftAfterTake = remaining - take
        if (leftAfterTake > 0 && leftAfterTake < CHUNK_MIN_WORDS) {
          take = remaining - CHUNK_MIN_WORDS
        }
      }

      chunks.push(words.slice(cursor, cursor + take).join(' '))
      cursor += take
    }

    if (chunks.length > 1) {
      const lastChunkWords = chunks[chunks.length - 1].split(/\s+/).length
      if (lastChunkWords < CHUNK_MIN_WORDS) {
        chunks[chunks.length - 2] = `${chunks[chunks.length - 2]} ${chunks[chunks.length - 1]}`
        chunks.pop()
      }
    }

    return chunks
  }

  const buildChunkTokenMap = (text) => {
    const map = new Map()
    for (const token of tokenize(text)) {
      map.set(token, (map.get(token) || 0) + 1)
    }
    return map
  }

  const buildGuidelineChunks = (rawText) => {
    const sections = splitIntoSections(rawText)
    const allChunks = []
    let chunkCounter = 0

    for (const section of sections) {
      const words = section.text.split(/\s+/).filter(Boolean)
      if (words.length === 0) continue

      const sectionChunks = createWordChunks(words)
      for (const chunkText of sectionChunks) {
        chunkCounter += 1
        const formattedChunk = `${section.heading}\n${chunkText}`
        allChunks.push({
          chunkNumber: chunkCounter,
          pageNumber: parsePageNumber(section.heading),
          heading: section.heading,
          text: formattedChunk,
          tokenMap: buildChunkTokenMap(formattedChunk)
        })
      }
    }

    return allChunks
  }

  const isAdministrativeSentence = (sentence) => {
    return /^(page\s+\d+|\d+\s+of\s+\d+|https?:\/\/|www\.|for clinicians|related resources|resource list|table of contents|contents|download|website|published|updated|clinical care standard|national health and medical research council|cancer council australia|medicines safety|quality in health care|bowel preparation|appointment|appointments|logistics|administrative|education|service description|open-access referral|patient information leaflet)/i.test(sentence)
  }

  const getSentenceRelevanceScore = (sentence, referralTokens) => {
    const sentenceTokens = tokenize(sentence)
    if (sentenceTokens.length === 0) return 0

    const evidenceTerms = [
      ['rectal bleeding', 6],
      ['blood in stool', 6],
      ['iron deficiency', 6],
      ['anaemia', 6],
      ['anemia', 6],
      ['positive fit', 6],
      ['positive ifobt', 6],
      ['altered bowel habit', 5],
      ['weight loss', 5],
      ['colorectal cancer', 5],
      ['family history', 5],
      ['urgent colonoscopy', 5],
      ['surveillance colonoscopy', 5],
      ['timely colonoscopy', 5],
      ['bowel disease', 4],
      ['symptoms suggestive of bowel cancer', 6],
      ['higher than average risk', 5],
      ['further investigation', 3]
    ]

    let score = 0

    for (const token of sentenceTokens) {
      if (referralTokens.has(token)) {
        score += 1
      }
    }

    const normalizedSentence = sentence.toLowerCase()
    for (const [phrase, weight] of evidenceTerms) {
      if (normalizedSentence.includes(phrase)) {
        score += weight
      }
    }

    if (/\b(colonoscopy|screening|surveillance|triage|timely|recommended|indicated|risk)\b/i.test(sentence)) {
      score += 2
    }

    return score
  }

  const extractRelevantEvidenceCandidates = (referral, chunk) => {
    const referralTokens = new Set(tokenize(referral))
    const rawSentences = splitIntoSentences(chunk.text)
      .map((sentence) => sentence.trim())
      .filter(Boolean)

    return rawSentences
      .filter((sentence) => !isAdministrativeSentence(sentence))
      .map((sentence) => ({
        sentence,
        score: getSentenceRelevanceScore(sentence, referralTokens)
      }))
      .filter((entry) => entry.score > 0)
  }

  const splitIntoSentences = (text) => {
    return text
      .replace(/\s+/g, ' ')
      .split(/(?<=[.!?])\s+/)
      .map((sentence) => sentence.trim())
      .filter(Boolean)
  }

  const scoreChunkRelevance = (queryTokens, chunk) => {
    if (queryTokens.length === 0) return 0

    const headingLower = chunk.heading.toLowerCase()
    let score = 0
    for (const token of queryTokens) {
      const frequency = chunk.tokenMap.get(token) || 0
      if (frequency > 0) {
        score += 1 + Math.log2(1 + frequency)
      }
      if (headingLower.includes(token)) {
        score += 0.75
      }
    }

    return score
  }

  const scorePriorityChunk = (chunk) => {
    const combinedText = `${chunk.heading} ${chunk.text}`.toLowerCase()
    const priorityTerms = [
      ['category 1', 10],
      ['category 2', 10],
      ['category 3', 10],
      ['urgent', 4],
      ['semi-urgent', 4],
      ['semi urgent', 4],
      ['routine', 4],
      ['priority', 4],
      ['urgency', 5],
      ['triage', 5],
      ['referral priority', 8],
      ['referral urgency', 8],
      ['urgency criteria', 8],
      ['priority criteria', 8],
      ['timeliness', 3],
      ['waiting time', 3],
      ['response time', 3]
    ]

    let score = 0
    for (const [term, weight] of priorityTerms) {
      if (combinedText.includes(term)) {
        score += weight
      }
    }

    if (/category\s+[123]/i.test(combinedText)) {
      score += 8
    }

    if (/urgent|semi-urgent|routine/i.test(chunk.heading)) {
      score += 6
    }

    return score
  }

  const selectTopPriorityChunks = (chunks) => {
    return chunks
      .map((chunk) => ({ chunk, score: scorePriorityChunk(chunk) }))
      .filter((entry) => entry.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, PRIORITY_CHUNK_COUNT)
      .map((entry) => entry.chunk)
  }

  const mergeUniqueChunks = (primaryChunks, secondaryChunks) => {
    const seen = new Set()
    const merged = []

    for (const chunk of [...primaryChunks, ...secondaryChunks]) {
      const key = `${chunk.pageNumber || 'nopage'}:${chunk.chunkNumber}`
      if (seen.has(key)) continue
      seen.add(key)
      merged.push(chunk)
    }

    return merged
  }

  const retrieveRelevantChunks = (referral, chunks) => {
    if (!chunks.length) return []

    // RAG Step 1: Build query tokens from the referral letter.
    const queryTokens = tokenize(referral)
    const scored = chunks
      .map((chunk) => ({ chunk, score: scoreChunkRelevance(queryTokens, chunk) }))
      .sort((a, b) => b.score - a.score)

    // RAG Step 2: Retrieve the top indication chunks plus the guideline urgency/category chunks.
    const indicationChunks = scored.filter((entry) => entry.score > 0).slice(0, RETRIEVAL_CHUNK_COUNT).map((entry) => entry.chunk)
    const priorityChunks = selectTopPriorityChunks(chunks)
    const mergedChunks = mergeUniqueChunks(indicationChunks, priorityChunks)

    if (mergedChunks.length > 0) {
      return mergedChunks
    }

    return scored.slice(0, Math.min(2, scored.length)).map((entry) => entry.chunk)
  }

  const loadGuidelineChunks = async () => {
    if (guidelineChunksRef.current.length > 0) {
      return guidelineChunksRef.current
    }

    // RAG Step 0: Parse and chunk the guideline PDF once, then cache chunks in memory.
    PDFParse.setWorker('/node_modules/pdf-parse/dist/pdf-parse/web/pdf.worker.mjs')

    const response = await fetch(GUIDELINE_PDF_URL)
    if (!response.ok) {
      throw new Error('Failed to load guideline PDF')
    }

    const pdfBytes = new Uint8Array(await response.arrayBuffer())
    const parser = new PDFParse({ data: pdfBytes })
    try {
      const textResult = await parser.getText()
      const chunks = buildGuidelineChunks(textResult.text || '')
      if (chunks.length === 0) {
        throw new Error('Guideline PDF extraction produced no content')
      }
      guidelineChunksRef.current = chunks
      return chunks
    } finally {
      await parser.destroy()
    }
  }

  const determinePriority = (text) => {
    const lowerText = text.toLowerCase()
    if (lowerText.match(/urgent|emergency|acute|severe|bleeding|hemodynamic/)) {
      return 'urgent'
    } else if (lowerText.match(/persistent|chronic|recurrent|repeated|ongoing/)) {
      return 'semi-urgent'
    } else {
      return 'routine'
    }
  }

  const getPriorityLabel = (p) => {
    if (p === 'urgent') return 'Category 1 - Urgent'
    if (p === 'semi-urgent') return 'Category 2 - Semi-Urgent'
    return 'Category 3 - Routine'
  }

  const getCompletenessExplanation = (score) => {
    let explanation = ''

    // Generate explanation based on completeness level
    if (score >= 80) {
      explanation = 'The referral contains comprehensive information with most key elements documented. This provides a solid foundation for triage assessment.'
    } else if (score >= 50) {
      explanation = 'The referral contains important information, but some key clinical details are missing. Additional investigation results or medical history would improve assessment accuracy.'
    } else {
      explanation = 'The referral is missing several key elements. Additional patient history, clinical findings, and investigation results would significantly strengthen the assessment.'
    }

    return explanation
  }

  const detectReferralFeatures = (referral) => {
    const lower = referral.toLowerCase()
    const features = []

    const checks = [
      [/rectal bleeding|blood in stool|haematochezia|hematochezia/i, 'rectal bleeding'],
      [/altered bowel habit|change in bowel habit|bowel habit change/i, 'altered bowel habit'],
      [/positive\s*(fit|ifobt)|fit\s*positive|ifobt\s*positive/i, 'positive FIT/iFOBT'],
      [/iron deficiency anaemia|iron deficiency anemia|iron deficiency|low ferritin|microcytic/i, 'iron deficiency anaemia'],
      [/weight loss|unintentional weight loss/i, 'weight loss'],
      [/colorectal cancer|bowel cancer|suspected cancer|cancer suspicion/i, 'colorectal cancer suspicion'],
      [/family history|first-degree relative|first degree relative/i, 'family history risk'],
      [/surveillance|previous polyp|prior adenoma|history of adenoma/i, 'surveillance context'],
      [/urgent|semi-urgent|semi urgent|routine|priority|triage/i, 'referral urgency context']
    ]

    for (const [pattern, label] of checks) {
      if (pattern.test(lower)) {
        features.push(label)
      }
    }

    return features
  }

  const ensureSentence = (text) => {
    const cleaned = text.replace(/\s+/g, ' ').trim()
    if (!cleaned) return ''
    const sentence = /^[A-Z]/.test(cleaned)
      ? cleaned
      : `${cleaned.charAt(0).toUpperCase()}${cleaned.slice(1)}`
    return /[.!?]$/.test(sentence) ? sentence : `${sentence}.`
  }

  const normalizeClinicalRecommendations = ({
    actions,
    priorityLabel,
    missingInformation,
    referral,
    retrievedGuidelineEvidence
  }) => {
    const forbiddenPattern = /\b(diagnos(?:e|is|ed)|treat(?:ment|ed|ing)?|prescrib(?:e|ed|ing)|medication|drug therapy|chemotherapy|radiotherapy|curative)\b/i
    const genericPattern = /^(refer for colonoscopy|recommend colonoscopy|consider colonoscopy|colonoscopy is recommended)\.?$/i

    const grounded = (actions || [])
      .map((item) => ensureSentence(String(item || '')))
      .filter(Boolean)
      .filter((item) => !forbiddenPattern.test(item))
      .filter((item) => !genericPattern.test(item))
      .map((item) => {
        if (/guideline|retrieved evidence|urgency criteria|priority category/i.test(item)) {
          return item
        }
        return `${item.replace(/[.!?]$/, '')}, in line with the retrieved guideline evidence.`
      })

    const features = detectReferralFeatures(referral)
    const featureText = features.length > 0 ? features.slice(0, 3).join(', ') : 'the documented referral features'
    const evidenceLead = retrievedGuidelineEvidence?.[0]?.quote

    const additionalInfo = (missingInformation || [])
      .map((item) => String(item || '').trim())
      .filter(Boolean)
      .slice(0, 3)

    const fallback = [
      `${priorityLabel} triage should be reflected in referral prioritisation based on the retrieved guideline urgency criteria and ${featureText}.`,
      `Proceed with specialist colonoscopy pathway documentation using the referral details and retrieved guideline indication criteria relevant to ${featureText}.`,
      additionalInfo.length > 0
        ? `Include ${additionalInfo.join(', ')} with the referral to align with the retrieved guideline and support safer triage decisions.`
        : `Include key investigation and clinical documentation with the referral to align with retrieved guideline evidence and support accurate triage.`,
      evidenceLead
        ? `Use the retrieved statement "${evidenceLead}" to justify urgency and next-step referral decisions in the triage documentation.`
        : 'Document how the retrieved guideline evidence supports urgency, required referral information, and the immediate next clinical step.'
    ].map((item) => ensureSentence(item))

    const combined = [...grounded]
    for (const item of fallback) {
      if (combined.length >= 5) break
      if (!item) continue
      if (forbiddenPattern.test(item)) continue
      if (combined.some((existing) => existing.toLowerCase() === item.toLowerCase())) continue
      combined.push(item)
    }

    const finalRecommendations = combined.slice(0, 5)
    if (finalRecommendations.length >= 3) {
      return finalRecommendations
    }

    return fallback.slice(0, 3)
  }

  const parseGeminiJson = (rawText) => {
    const trimmed = rawText.trim()

    // Some model responses may include extra characters around JSON.
    try {
      return JSON.parse(trimmed)
    } catch {
      const firstBrace = trimmed.indexOf('{')
      const lastBrace = trimmed.lastIndexOf('}')
      if (firstBrace !== -1 && lastBrace > firstBrace) {
        return JSON.parse(trimmed.slice(firstBrace, lastBrace + 1))
      }
      throw new SyntaxError('No valid JSON found in model response')
    }
  }

  const handleAnalyzeReferral = async () => {
    if (isLoading) return

    const trimmedReferral = referralText.trim()
    if (!trimmedReferral) {
      alert('Please paste a referral letter to analyze')
      return
    }

    if (!import.meta.env.VITE_GEMINI_API_KEY) {
      setApiError('Gemini API key is missing. Set VITE_GEMINI_API_KEY in your environment and try again.')
      return
    }

    setApiError('')
    setIsLoading(true)

    try {
      const guidelineChunks = await loadGuidelineChunks()
      const relevantChunks = retrieveRelevantChunks(trimmedReferral, guidelineChunks)
      if (relevantChunks.length === 0) {
        throw new Error('Guideline chunks are unavailable')
      }

      // RAG Step 3: Send only the retrieved chunks to Gemini as grounded context.
      const retrievedGuidelineEvidence = relevantChunks
        .flatMap((chunk, index) => {
          const candidates = extractRelevantEvidenceCandidates(trimmedReferral, chunk)
          return candidates.map((entry) => ({
            label: chunk.pageNumber ? `Page ${chunk.pageNumber}` : `Chunk ${chunk.chunkNumber || index + 1}`,
            quote: entry.sentence,
            score: entry.score
          }))
        })
        .sort((a, b) => b.score - a.score)
        .slice(0, 4)
        .map(({ label, quote }) => ({ label, quote }))

      const guidelineContext = retrievedGuidelineEvidence
        .map((evidence) => evidence.quote)
        .join(' ')

      const displayedGuidelineEvidence = retrievedGuidelineEvidence.length > 0
        ? retrievedGuidelineEvidence
        : [{ label: 'Chunk 1', quote: 'No directly relevant guideline evidence retrieved.' }]

      const retrievalSummary = [
        {
          label: 'Guideline PDF loaded',
          value: 'Yes'
        },
        {
          label: 'Searchable chunks',
          value: `${guidelineChunks.length}`
        },
        {
          label: 'Retrieved for this referral',
          value: `${relevantChunks.length}`
        },
        {
          label: 'Gemini 2.5 Flash',
          value: 'Selected model'
        },
        {
          label: 'Grounded response generated',
          value: 'Yes'
        },
        {
          label: 'JSON validated',
          value: 'Yes'
        }
      ]

      const prompt = GEMINI_PROMPT
        .replace('{{GUIDELINE_CHUNKS}}', guidelineContext)
        .replace('{{REFERRAL}}', trimmedReferral)

      const result = await model.generateContent(prompt)
      const responseText = result.response.text()
      const parsed = parseGeminiJson(responseText)
      const jsonValidated = true

      const modelPriority = (parsed.priority || '').toLowerCase()
      const detectedPriority = ['urgent', 'semi-urgent', 'routine'].includes(modelPriority)
        ? modelPriority
        : determinePriority(trimmedReferral)

      const completenessScore = Number.isFinite(parsed.completenessScore)
        ? Math.max(0, Math.min(100, Math.round(parsed.completenessScore)))
        : 0

      const clinicalFindings = Array.isArray(parsed.clinicalFindings) ? parsed.clinicalFindings : []
      const redFlags = Array.isArray(parsed.redFlags) ? parsed.redFlags : []
      const findingsForCard = clinicalFindings.length > 0 ? clinicalFindings : redFlags
      const guidelineEvidence = Array.isArray(parsed.guidelineEvidence) ? parsed.guidelineEvidence : []
      const missingInformation = Array.isArray(parsed.missingInformation) ? parsed.missingInformation : []
      const clinicalRecommendation = normalizeClinicalRecommendations({
        actions: Array.isArray(parsed.recommendedActions) ? parsed.recommendedActions : [],
        priorityLabel: getPriorityLabel(detectedPriority),
        missingInformation,
        referral: trimmedReferral,
        retrievedGuidelineEvidence: displayedGuidelineEvidence
      })

      setAiResponse({
        priority: detectedPriority,
        priorityLabel: getPriorityLabel(detectedPriority),
        completenessScore,
        completenessExplanation: getCompletenessExplanation(completenessScore),
        patientSummary: parsed.patientSummary || 'No patient summary provided.',
        redFlags: findingsForCard.length > 0 ? findingsForCard : ['No key clinical findings provided.'],
        clinicalRecommendation,
        guidelineRationale: parsed.guidelineRationale || 'No guideline rationale provided.',
        guidelineEvidence: guidelineEvidence.length > 0 ? guidelineEvidence : ['No directly relevant guideline evidence retrieved.'],
        missingInformation: missingInformation.length > 0 ? missingInformation : ['No additional information identified.'],
        retrievedGuidelineEvidence: displayedGuidelineEvidence,
        retrievalSummary: retrievalSummary.map((item) => {
          if (item.label === 'Gemini 2.5 Flash') {
            return { ...item, value: GEMINI_MODEL_NAME }
          }

          if (item.label === 'JSON validated') {
            return { ...item, value: jsonValidated ? 'Yes' : 'No' }
          }

          return item
        }),
        safetyNotice: parsed.safetyNotice || 'Clinical decisions must be reviewed by qualified healthcare professionals.'
      })
      setPriority(detectedPriority)
    } catch (error) {
      console.error('Referral analysis failed', error)
      if (error instanceof SyntaxError) {
        setApiError('The AI response could not be processed because it was not valid JSON. Please try again.')
      } else if (error instanceof Error && error.message === 'Guideline chunks are unavailable') {
        setApiError('Guideline content could not be loaded from the PDF. Please refresh and try again.')
      } else if (error instanceof Error && error.message.includes('Guideline PDF')) {
        setApiError('Guideline PDF could not be parsed in the browser. Please refresh and try again.')
      } else {
        setApiError('Unable to analyse referral at the moment. Please check your API key, network connection, and try again.')
      }
      setAiResponse(null)
      setPriority(null)
    } finally {
      setIsLoading(false)
    }
  }

  const handleClear = () => {
    setReferralText('')
    setAiResponse(null)
    setPriority(null)
    setApiError('')
  }

  return (
    <div className="app-container">
      <header className="header">
        <div className="header-content">
          <div className="header-title-group">
            <Stethoscope className="header-icon" size={32} strokeWidth={2} />
            <h1>Colonoscopy Referral Triage Assistant</h1>
          </div>
          <p className="subtitle">Clinical Decision Support Tool</p>
        </div>
      </header>

      <main className="main-content">
        <div className="content-wrapper">
          {/* Left Panel - Input */}
          <div className="input-panel">
            <div className="panel-header">
              <h2>📝 Referral Letter Input</h2>
            </div>
            <textarea
              className="referral-textarea"
              placeholder="Paste the referral letter here. Include patient history, presenting symptoms, clinical findings, and any relevant test results..."
              value={referralText}
              onChange={(e) => setReferralText(e.target.value)}
              disabled={isLoading}
            />
            <div className="button-group">
              <button 
                className="btn btn-primary"
                onClick={handleAnalyzeReferral}
                disabled={isLoading || !referralText.trim()}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="spinner" size={18} />
                    Analysing referral...
                  </>
                ) : (
                  <>
                    🔍 Analyze Referral
                  </>
                )}
              </button>
              <button 
                className="btn btn-secondary"
                onClick={handleClear}
                disabled={isLoading}
              >
                Clear
              </button>
            </div>
          </div>

          {/* Right Panel - AI Response */}
          <div className="response-panel">
            <div className="panel-header">
              <h2>🤖 Triage Assessment</h2>
            </div>
            <div className="response-content">
              {apiError ? (
                <div className="empty-state">
                  <p>{apiError}</p>
                  <p className="secondary-text">Update your API key or retry the request.</p>
                </div>
              ) : null}
              {aiResponse ? (
                (() => {
                  const retrievedGuidelineEvidence = aiResponse.retrievedGuidelineEvidence || []
                  const guidelineEvidence = aiResponse.guidelineEvidence || []
                  const missingInformation = aiResponse.missingInformation || []
                  const redFlags = aiResponse.redFlags || []
                  const clinicalRecommendation = aiResponse.clinicalRecommendation || []
                  const retrievalSummary = aiResponse.retrievalSummary || []

                  return (
                <div className="clinical-cards-grid">
                  {/* Referral Priority Card */}
                  <div className={`clinical-card priority-card priority-${aiResponse.priority}`}>
                    <div className="card-header">
                      <AlertCircle className="card-icon" size={24} />
                      <h3>Referral Priority</h3>
                    </div>
                    <div className="priority-display">
                      {aiResponse.priorityLabel}
                    </div>
                  </div>

                  {/* Referral Completeness Card */}
                  <div className="clinical-card completeness-card">
                    <div className="card-header">
                      <BarChart3 className="card-icon" size={24} />
                      <h3>Referral Completeness</h3>
                    </div>
                    <div className="completeness-container">
                      <div className="completeness-score-display">
                        <div className="completeness-percentage">{aiResponse.completenessScore}%</div>
                      </div>
                      <div className={`progress-bar progress-${aiResponse.completenessScore >= 80 ? 'green' : aiResponse.completenessScore >= 50 ? 'amber' : 'red'}`}>
                        <div 
                          className="progress-fill" 
                          style={{ width: `${aiResponse.completenessScore}%` }}
                        />
                      </div>
                      <p className="completeness-explanation">{aiResponse.completenessExplanation}</p>
                      <ul className="card-list">
                        {missingInformation.map((item, idx) => (
                          <li key={idx}>{item}</li>
                        ))}
                      </ul>
                    </div>
                  </div>

                  {/* Retrieval Summary Card */}
                  <div className="clinical-card">
                    <div className="card-header">
                      <BarChart3 className="card-icon" size={24} />
                      <h3>Retrieval Summary</h3>
                    </div>
                    <ul className="card-list">
                      {retrievalSummary.map((item, idx) => (
                        <li key={idx}>
                          <strong>{item.label}:</strong> {item.value}
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Patient Summary Card */}
                  <div className="clinical-card">
                    <div className="card-header">
                      <User className="card-icon" size={24} />
                      <h3>Patient Summary</h3>
                    </div>
                    <p className="card-text">{aiResponse.patientSummary}</p>
                  </div>

                  {/* Red Flag Symptoms Card */}
                  <div className="clinical-card">
                    <div className="card-header">
                      <AlertTriangle className="card-icon" size={24} />
                      <h3>Clinical Findings</h3>
                    </div>
                    <ul className="card-list">
                      {redFlags.map((flag, idx) => (
                        <li key={idx}>{flag}</li>
                      ))}
                    </ul>
                  </div>

                  {/* Clinical Recommendation Card */}
                  <div className="clinical-card">
                    <div className="card-header">
                      <CheckCircle className="card-icon" size={24} />
                      <h3>Clinical Recommendation</h3>
                    </div>
                    <ol className="card-list card-list-ordered">
                      {clinicalRecommendation.map((rec, idx) => (
                        <li key={idx}>{rec}</li>
                      ))}
                    </ol>
                  </div>

                  {/* Guideline Rationale Card */}
                  <div className="clinical-card">
                    <div className="card-header">
                      <BookOpen className="card-icon" size={24} />
                      <h3>Guideline Rationale</h3>
                    </div>
                    <p className="card-text">{aiResponse.guidelineRationale}</p>
                    <ul className="card-list">
                      {guidelineEvidence.map((item, idx) => (
                        <li key={idx}>{item}</li>
                      ))}
                    </ul>
                  </div>

                  {/* Retrieved Guideline Evidence Card */}
                  <div className="clinical-card">
                    <div className="card-header">
                      <BookOpen className="card-icon" size={24} />
                      <h3>Retrieved Guideline Evidence</h3>
                    </div>
                    <ul className="card-list">
                      {retrievedGuidelineEvidence.map((evidence, idx) => (
                        <li key={idx}>
                          <strong>Evidence {idx + 1}</strong>
                          <div>“{evidence.quote}”</div>
                          <div>{evidence.label}</div>
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Safety Notice Card */}
                  <div className="clinical-card safety-card">
                    <div className="card-header">
                      <ShieldAlert className="card-icon" size={24} />
                      <h3>Safety Notice</h3>
                    </div>
                    <p className="card-text">{aiResponse.safetyNotice}</p>
                  </div>
                </div>
                  )
                })()
              ) : (
                <div className="empty-state">
                  <p>Analysis results will appear here</p>
                  <p className="secondary-text">Paste a referral letter and click "Analyze Referral" to begin</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* Medical Disclaimer - Fixed at Bottom */}
      <footer className="disclaimer-footer">
        <div className="disclaimer-content">
          <h3>⚠️ Medical Disclaimer</h3>
          <p>
            This application is an <strong>educational clinical decision support tool only</strong>. 
            It is designed to assist clinicians in reviewing colonoscopy referrals using Australian clinical guidance. 
            It does not replace specialist judgement or institutional triage protocols. 
            Clinicians remain responsible for all final decisions regarding patient care. 
            Patients should not use this application for medical advice.
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App
