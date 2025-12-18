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
import java.util.Optional
import com.google.common.collect.ImmutableList
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInference.LlmInferenceOptions
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.guava.await
import kotlin.math.max

class RagPipeline(private val app: Application) {

    private val llmOptions: LlmInferenceOptions =
        LlmInferenceOptions.builder()
            .setModelPath(GEMMA_MODEL_PATH)
            .setPreferredBackend(LlmInference.Backend.CPU)
            .setMaxTokens(1024)
            .build()

    private val sessionOptions: LlmInferenceSession.LlmInferenceSessionOptions =
        LlmInferenceSession.LlmInferenceSessionOptions.builder()
            .setTemperature(0.7f)
            .setTopP(0.95f)
            .setTopK(64)
            .build()

    private val languageModel = MediaPipeLlmBackend(app.applicationContext, llmOptions, sessionOptions)
    private val modelInitFuture = languageModel.initialize()

    private val embedder: Embedder<String> =
        GeckoEmbeddingModel(
            EMBEDDING_MODEL_PATH,
            Optional.of(TOKENIZER_MODEL_PATH),
            USE_GPU_FOR_EMBEDDINGS
        )

    private val config = ChainConfig.create(
        languageModel,
        PromptBuilder(QA_PROMPT_TEMPLATE),
        DefaultSemanticTextMemory(
            // DefaultVectorStore tracks the dimensionality internally now.
            DefaultVectorStore(),
            embedder
        )
    )

    private val chain = RetrievalAndInferenceChain(config)

    suspend fun awaitModelReady() {
        modelInitFuture.await()
    }

    /**
     * Index user text with recursive chunking + overlap.
     * Enforces <= 245 tokens per chunk to stay under 256-token embedder limit.
     *
     * The official guide notes embedding models truncate beyond their max sequence length. :contentReference[oaicite:3]{index=3}
     */
    suspend fun indexUserText(
        raw: String,
        onChunkCountKnown: (Int) -> Unit = {}
    ) {
        val chunks = RecursiveChunker.chunk(
            text = raw,
            maxTokens = MAX_TOKENS_PER_CHUNK,
            overlapTokens = OVERLAP_TOKENS
        )
        onChunkCountKnown(chunks.size)

        val semanticMemoryOpt = config.semanticMemory
        val semanticMemory = if (semanticMemoryOpt.isPresent) semanticMemoryOpt.get() else null
        if (semanticMemory != null && chunks.isNotEmpty()) {
            semanticMemory.recordBatchedMemoryItems(ImmutableList.copyOf(chunks)).await()
        }
    }

    suspend fun generateResponse(
        prompt: String,
        onPartial: (String) -> Unit
    ): String = coroutineScope {
        val request = RetrievalRequest.create(
            prompt,
            RetrievalConfig.create(
                TOP_K,
                0.0f,
                TaskType.QUESTION_ANSWERING
            )
        )

        val listener = AsyncProgressListener<LanguageModelResponse> { resp, _ ->
            onPartial(resp.text)
        }

        chain.invoke(request, listener).await().text
    }

    private object RecursiveChunker {
        private val seps = listOf("\n\n", "\n", ". ", " ")

        fun chunk(text: String, maxTokens: Int, overlapTokens: Int): List<String> {
            val cleaned = text.replace("\u0000", "").trim()
            if (cleaned.isEmpty()) return emptyList()

            val base = splitRec(cleaned, maxTokens, 0)
                .map { it.trim() }
                .filter { it.isNotEmpty() }

            // Add overlap at token level
            if (base.isEmpty()) return emptyList()
            if (overlapTokens <= 0) return base.map { capTokens(it, maxTokens) }

            val out = ArrayList<String>(base.size)
            var prevTailTokens: List<String> = emptyList()

            for (piece in base) {
                val pieceTokens = tokenize(piece)
                val withOverlapTokens =
                    if (prevTailTokens.isEmpty()) pieceTokens
                    else (prevTailTokens + pieceTokens).take(maxTokens)

                out.add(withOverlapTokens.joinToString(" "))

                val tailCount = max(0, overlapTokens)
                prevTailTokens = pieceTokens.takeLast(minOf(tailCount, pieceTokens.size))
            }

            return out
        }

        private fun splitRec(text: String, maxTokens: Int, sepIdx: Int): List<String> {
            if (countTokens(text) <= maxTokens) return listOf(text)

            if (sepIdx >= seps.size) {
                return chunkByTokens(text, maxTokens)
            }

            val sep = seps[sepIdx]
            val parts = text.split(sep)
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
                if (countTokens(candidate) <= maxTokens) {
                    if (sb.isNotEmpty()) sb.append(sep)
                    sb.append(p)
                } else {
                    flush()
                    // if single part still too big, recurse deeper
                    if (countTokens(p) > maxTokens) {
                        blocks.addAll(splitRec(p, maxTokens, sepIdx + 1))
                    } else {
                        sb.append(p)
                    }
                }
            }
            flush()

            return blocks.flatMap {
                if (countTokens(it) > maxTokens) splitRec(it, maxTokens, sepIdx + 1) else listOf(it)
            }.map { capTokens(it, maxTokens) }
        }

        private fun chunkByTokens(text: String, maxTokens: Int): List<String> {
            val t = tokenize(text)
            if (t.isEmpty()) return emptyList()
            val out = mutableListOf<String>()
            var i = 0
            while (i < t.size) {
                val end = minOf(i + maxTokens, t.size)
                out.add(t.subList(i, end).joinToString(" "))
                i = end
            }
            return out
        }

        private fun capTokens(text: String, maxTokens: Int): String {
            val t = tokenize(text)
            return if (t.size <= maxTokens) text.trim() else t.take(maxTokens).joinToString(" ")
        }

        private fun tokenize(text: String): List<String> =
            text.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }

        private fun countTokens(text: String): Int = tokenize(text).size
    }

    companion object {
private const val USE_GPU_FOR_EMBEDDINGS = false

        private const val MAX_TOKENS_PER_CHUNK = 245
        private const val OVERLAP_TOKENS = 32

        private const val TOP_K = 3

        // Put these on-device with adb (see steps)
        private const val GEMMA_MODEL_PATH = "/data/local/tmp/llm/gemma3-1b-it-int4.task"
        private const val TOKENIZER_MODEL_PATH = "/data/local/tmp/llm/sentencepiece.model"
        private const val EMBEDDING_MODEL_PATH =
            "/data/local/tmp/llm/embeddinggemma-300M_seq256_mixed-precision.tflite"

        private const val QA_PROMPT_TEMPLATE =
            "You are a helpful assistant. Use ONLY the provided context to answer.\n\n" +
                    "Context:\n{0}\n\n" +
                    "User question:\n{1}\n\n" +
                    "Answer:"
    }
}
