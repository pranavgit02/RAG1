package com.example.rag1

import android.app.Application
import com.google.ai.edge.localagents.rag.chains.ChainConfig
import com.google.ai.edge.localagents.rag.chains.RetrievalAndInferenceChain
import com.google.ai.edge.localagents.rag.memory.DefaultSemanticTextMemory
import com.google.ai.edge.localagents.rag.memory.DefaultVectorStore
import com.google.ai.edge.localagents.rag.models.AsyncProgressListener
import com.google.ai.edge.localagents.rag.models.Embedder
import com.google.ai.edge.localagents.rag.models.GeckoEmbeddingModel
import com.google.ai.edge.localagents.rag.models.LanguageModelResponse
import com.google.ai.edge.localagents.rag.models.MediaPipeLlmBackend
import com.google.ai.edge.localagents.rag.prompt.PromptBuilder
import com.google.ai.edge.localagents.rag.retrieval.RetrievalConfig
import com.google.ai.edge.localagents.rag.retrieval.RetrievalConfig.TaskType
import com.google.ai.edge.localagents.rag.retrieval.RetrievalRequest
import com.google.common.collect.ImmutableList
import com.google.mediapipe.tasks.genai.llminference.GraphOptions
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInference.LlmInferenceOptions
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.guava.await
import java.io.File
import java.util.Optional
import kotlin.jvm.optionals.getOrNull
import kotlin.math.max

private const val ENABLE_VISION = true

class RagPipeline(private val app: Application) {

    // Prefer app-internal storage (what the reference app uses), but keep legacy /data/local/tmp fallback.
    private val llmModelFilePrimary = File(app.filesDir, "llm/$GEMMA_TASK_FILENAME")
    private val llmModelFileLegacy = File("/data/local/tmp/llm/$GEMMA_TASK_FILENAME")

    private val embedModelFilePrimary = File(app.filesDir, "llm/$EMBEDDING_TFLITE_FILENAME")
    private val embedModelFileLegacy = File("/data/local/tmp/llm/$EMBEDDING_TFLITE_FILENAME")

    private val tokenizerFilePrimary = File(app.filesDir, "llm/$TOKENIZER_FILENAME")
    private val tokenizerFileLegacy = File("/data/local/tmp/llm/$TOKENIZER_FILENAME")

    private fun pickExistingPath(primary: File, legacy: File): String =
        when {
            primary.exists() -> primary.absolutePath
            legacy.exists() -> legacy.absolutePath
            else -> primary.absolutePath // default (so you can later download/copy into this location)
        }

    private val llmModelPath: String = pickExistingPath(llmModelFilePrimary, llmModelFileLegacy)
    private val embeddingModelPath: String = pickExistingPath(embedModelFilePrimary, embedModelFileLegacy)
    private val tokenizerModelPath: String = pickExistingPath(tokenizerFilePrimary, tokenizerFileLegacy)

    private val llmOptions: LlmInferenceOptions =
        LlmInferenceOptions.builder()
            .setModelPath(llmModelPath)
            .setPreferredBackend(LlmInference.Backend.CPU)
            .setMaxTokens(1024)
            .setMaxNumImages(if (ENABLE_VISION) 1 else 0)
            .build()

    private val sessionOptions: LlmInferenceSession.LlmInferenceSessionOptions =
        LlmInferenceSession.LlmInferenceSessionOptions.builder()
            .setTemperature(0.7f)
            .setTopP(0.95f)
            .setTopK(64)
            .setGraphOptions(
                GraphOptions.builder()
                    .setEnableVisionModality(ENABLE_VISION)
                    .build()
            )
            .build()

    private val languageModel = MediaPipeLlmBackend(app.applicationContext, llmOptions, sessionOptions)

    // IMPORTANT: initialize() completing does NOT always mean inference is usable yet.
    // Some builds return "LLM inference is not initialized yet!" for a while after init.
    private val modelInitFuture = languageModel.initialize()

    @Volatile private var didWarmup: Boolean = false

    /** Returns true if the LLM .task file path looks present on-device (primary or legacy). */
    fun isLlmModelPresent(): Boolean = llmModelFilePrimary.exists() || llmModelFileLegacy.exists()

    /** Returns true if the embedder model + tokenizer look present on-device (primary or legacy). */
    fun isEmbedderModelPresent(): Boolean =
        (embedModelFilePrimary.exists() || embedModelFileLegacy.exists()) &&
                (tokenizerFilePrimary.exists() || tokenizerFileLegacy.exists())

    suspend fun awaitModelReady() {
        modelInitFuture.await()
    }

    /** Backwards-compatible alias (your UI can call this). */
    suspend fun awaitLlmReady() {
        awaitModelReady()
    }

    suspend fun warmupLlm() {
        if (didWarmup) return

        // Wait for initialize() future (even if it doesn't throw, we still verify below)
        awaitModelReady()

        val request =
            RetrievalRequest.create(
                "Reply with a single word: OK.",
                RetrievalConfig.create(
                    /*topK=*/ 1,
                    /*minSimilarity=*/ 0.0f,
                    TaskType.QUESTION_ANSWERING
                )
            )

        val result =
            chain.invoke(
                request,
                AsyncProgressListener<LanguageModelResponse> { _, _ -> }
            ).await()

        val text = result.text.trim()

        // IMPORTANT: detect the placeholder that means engine isn't usable
        if (text.contains("not initialized", ignoreCase = true)) {
            throw IllegalStateException(
                "LLM warmup failed (engine not usable). Got placeholder: '$text'. " +
                        "Check logcat for the real init error."
            )
        }

        didWarmup = true
    }

    private val embedder: Embedder<String> =
        GeckoEmbeddingModel(
            embeddingModelPath,
            Optional.of(tokenizerModelPath),
            USE_GPU_FOR_EMBEDDINGS
        )

    private val config =
        ChainConfig.create(
            languageModel,
            PromptBuilder(QA_PROMPT_TEMPLATE),
            DefaultSemanticTextMemory(
                DefaultVectorStore(),
                embedder
            )
        )

    private val chain = RetrievalAndInferenceChain(config)

    suspend fun indexUserText(
        raw: String,
        onChunkCountKnown: (Int) -> Unit = {}
    ) {
        val chunks =
            RecursiveChunker.chunk(
                text = raw,
                maxTokens = MAX_TOKENS_PER_CHUNK,
                overlapTokens = OVERLAP_TOKENS
            )
        onChunkCountKnown(chunks.size)

        val semanticMemory = config.semanticMemory.getOrNull() ?: return
        if (chunks.isEmpty()) return

        suspend fun recordSafely(chunk: String, depth: Int = 0) {
            if (depth >= 10) throw IllegalStateException("Chunk still too large after repeated splits.")

            try {
                semanticMemory.recordBatchedMemoryItems(ImmutableList.of(chunk)).await()
            } catch (t: Throwable) {
                val msg = (t.message ?: "").lowercase()
                val tooLong =
                    msg.contains("max_input_size") ||
                            msg.contains("max input size") ||
                            msg.contains("tokens.size()") ||
                            msg.contains("sequence length")

                if (tooLong && chunk.length > 1) {
                    val (a, b) = splitNearMiddle(chunk)
                    if (a.isNotBlank()) recordSafely(a, depth + 1)
                    if (b.isNotBlank()) recordSafely(b, depth + 1)
                } else {
                    throw t
                }
            }
        }

        for (c in chunks) recordSafely(c)
    }

    private fun splitNearMiddle(text: String): Pair<String, String> {
        val cleaned = text.trim()
        if (cleaned.length < 2) return cleaned to ""
        val mid = cleaned.length / 2

        val left = cleaned.lastIndexOf(' ', startIndex = mid)
        val right = cleaned.indexOf(' ', startIndex = mid)

        val splitAt =
            when {
                left >= 0 && right >= 0 -> if (mid - left <= right - mid) left else right
                left >= 0 -> left
                right >= 0 -> right
                else -> mid
            }

        val a = cleaned.substring(0, splitAt).trim()
        val b = cleaned.substring(splitAt).trim()
        return a to b
    }

    private fun isNotInitializedText(text: String?): Boolean {
        val t = text?.trim().orEmpty()
        if (t.isEmpty()) return false

        // Catch the placeholder response you were seeing
        return t.contains("LLM inference is not initialized yet", ignoreCase = true) ||
                t.contains("not initialized", ignoreCase = true)
    }

    suspend fun generateResponse(
        prompt: String,
        onPartial: (String) -> Unit
    ): String =
        coroutineScope {
            // Make sure we never try inference until warmup has actually succeeded.
            if (!didWarmup) warmupLlm()

            val request =
                RetrievalRequest.create(
                    prompt,
                    RetrievalConfig.create(
                        TOP_K,
                        0.0f,
                        TaskType.QUESTION_ANSWERING
                    )
                )

            val listener =
                AsyncProgressListener<LanguageModelResponse> { resp, _ ->
                    // Avoid spamming the UI with the placeholder text.
                    if (!isNotInitializedText(resp.text)) onPartial(resp.text)
                }

            // Sometimes the first request right after warmup can still race.
            // Retry a couple times if we get the placeholder.
            var lastText: String? = null
            repeat(3) { attemptIdx ->
                val resp = chain.invoke(request, listener).await()
                val text = resp.text?.trim()
                lastText = text

                if (!isNotInitializedText(text)) return@coroutineScope (text ?: "")

                if (attemptIdx < 2) delay(500L)
            }

            throw IllegalStateException(lastText?.takeIf { it.isNotBlank() } ?: "LLM inference is not initialized yet!")
        }

    private object RecursiveChunker {
        private val seps = listOf("\n\n", "\n", ". ", " ")

        fun chunk(text: String, maxTokens: Int, overlapTokens: Int): List<String> {
            val cleaned = text.trim()
            if (cleaned.isEmpty()) return emptyList()

            fun approxTokens(s: String): Int {
                // rough heuristic (safe for "keep under 245 tokens" goal)
                val words = s.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }.size
                return max(1, (words * 1.35f).toInt())
            }

            fun splitRec(text: String, maxTokens: Int, sepIdx: Int): List<String> {
                if (approxTokens(text) <= maxTokens) return listOf(text.trim())
                if (sepIdx >= seps.size) {
                    // fallback: hard split by characters
                    val n = max(1, maxTokens * 4)
                    return text.chunked(n).map { it.trim() }.filter { it.isNotEmpty() }
                }

                val sep = seps[sepIdx]
                val parts =
                    text.split(sep)
                        .map { it.trim() }
                        .filter { it.isNotEmpty() }

                if (parts.size <= 1) return splitRec(text, maxTokens, sepIdx + 1)

                // Greedy re-pack parts into <= maxTokens blocks
                val blocks = mutableListOf<String>()
                val sb = StringBuilder()

                fun flush() {
                    val s = sb.toString().trim()
                    if (s.isNotEmpty()) blocks.add(s)
                    sb.clear()
                }

                for (p in parts) {
                    val candidate = if (sb.isEmpty()) p else sb.toString() + sep + p
                    if (approxTokens(candidate) <= maxTokens) {
                        sb.clear()
                        sb.append(candidate)
                    } else {
                        flush()
                        if (approxTokens(p) <= maxTokens) {
                            sb.append(p)
                        } else {
                            blocks.addAll(splitRec(p, maxTokens, sepIdx + 1))
                        }
                    }
                }
                flush()
                return blocks
            }

            val baseChunks = splitRec(cleaned, maxTokens, 0)

            // Add overlap
            if (overlapTokens <= 0 || baseChunks.size <= 1) return baseChunks

            val out = mutableListOf<String>()
            for (i in baseChunks.indices) {
                val cur = baseChunks[i]
                val prev = if (i > 0) baseChunks[i - 1] else ""
                if (prev.isBlank()) {
                    out.add(cur)
                    continue
                }
                val prevWords = prev.split(Regex("\\s+")).filter { it.isNotEmpty() }
                val take = max(0, minOf(prevWords.size, overlapTokens))
                val overlap =
                    if (take == 0) ""
                    else prevWords.takeLast(take).joinToString(" ")

                out.add((overlap + "\n" + cur).trim())
            }
            return out
        }
    }

    private companion object {
        private const val USE_GPU_FOR_EMBEDDINGS = false

        // Keep below 245 tokens (embedder limit is 256 seq; keep safety margin)
        private const val MAX_TOKENS_PER_CHUNK = 235
        private const val OVERLAP_TOKENS = 20
        private const val TOP_K = 3

        // Filenames (actual path resolved at runtime: filesDir/llm/ or legacy /data/local/tmp/llm/)
        private const val GEMMA_TASK_FILENAME = "gemma-3n-E2B-it-int4.task"
        private const val TOKENIZER_FILENAME = "sentencepiece.model"
        private const val EMBEDDING_TFLITE_FILENAME = "embeddinggemma-300M_seq256_mixed-precision.tflite"

        private const val QA_PROMPT_TEMPLATE =
            "You are a helpful assistant.\n" +
                    "If Context is empty or not relevant, answer normally.\n" +
                    "If Context contains relevant info, prefer using it.\n\n" +
                    "Context:\n{0}\n\n" +
                    "User:\n{1}\n\n" +
                    "Assistant:"

    }
}
