package com.example.rag1

import android.app.Application
import android.net.Uri
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

enum class Role { User, Model }

data class ChatMessage(
    val role: Role,
    val text: String,
)

class ChatViewModel(app: Application) : AndroidViewModel(app) {

    private var pipeline = RagPipeline(app)
    private var llmInitJob: Job? = null

    val messages = mutableStateListOf<ChatMessage>()

    // Model status (separate embedder vs LLM)
    val embedderStatusText = mutableStateOf("initializing embedder...")
    val isEmbedderReady = mutableStateOf(false)

    val llmStatusText = mutableStateOf("initializing LLM...")
    val isLlmReady = mutableStateOf(false)

    val ragStatusText = mutableStateOf("No .txt loaded")
    val isIndexing = mutableStateOf(false)
    val isRagReady = mutableStateOf(false)
    val loadedFileName = mutableStateOf<String?>(null)

    init {
        startPipelineInit()
    }

    private fun startPipelineInit() {
        llmInitJob?.cancel()
        llmInitJob = null

        // --- Embedder status (only required for RAG/indexing) ---
        val embedderPresent = pipeline.isEmbedderModelPresent()
        isEmbedderReady.value = embedderPresent
        embedderStatusText.value = if (embedderPresent) "Embedder Ready" else "Embedder model missing"

        // --- LLM status (required for both normal chat and RAG) ---
        val llmPresent = pipeline.isLlmModelPresent()
        isLlmReady.value = false
        llmStatusText.value = if (llmPresent) "Initializing LLM..." else "LLM model missing"
        if (!llmPresent) return

        llmInitJob =
            viewModelScope.launch {
                try {
                    withContext(Dispatchers.Default) {
                        pipeline.awaitLlmReady()
                        // Warmup prevents false "ready" states.
                        pipeline.warmupLlm()
                    }
                    isLlmReady.value = true
                    llmStatusText.value = "LLM Model Ready"
                } catch (t: Throwable) {
                    isLlmReady.value = false
                    llmStatusText.value = "LLM init failed"
                    ragStatusText.value = "Error: ${t.message ?: t.javaClass.simpleName}"
                }
            }
    }

    /**
     * Behavior:
     * - If NO file attached: regular chat mode (LLM-only) => allow send when LLM ready and not indexing.
     * - If file attached: RAG mode => allow send only after indexing completes (isRagReady).
     */
    fun canSendNow(currentInput: String): Boolean {
        if (currentInput.isBlank()) return false
        if (!isLlmReady.value) return false
        if (isIndexing.value) return false

        val hasFile = loadedFileName.value != null
        if (hasFile && !isRagReady.value) return false

        return true
    }

    fun onPickTxt(uri: Uri, displayName: String?) {
        val ctx = getApplication<Application>().applicationContext
        viewModelScope.launch {
            isIndexing.value = true
            isRagReady.value = false
            loadedFileName.value = displayName ?: "selected_file.txt"
            ragStatusText.value = "Reading file..."

            if (!isEmbedderReady.value) {
                ragStatusText.value = "Embedder not ready"
                isIndexing.value = false
                return@launch
            }

            try {
                val rawText =
                    withContext(Dispatchers.IO) {
                        ctx.contentResolver.openInputStream(uri)?.use {
                            it.readBytes().toString(Charsets.UTF_8)
                        } ?: throw IllegalStateException("Could not open file")
                    }

                ragStatusText.value = "Chunking + indexing..."
                withContext(Dispatchers.Default) {
                    pipeline.indexUserText(rawText) { chunkCount ->
                        viewModelScope.launch(Dispatchers.Main) {
                            ragStatusText.value = "Indexing... ($chunkCount chunks)"
                        }
                    }
                }

                isRagReady.value = true
                ragStatusText.value = "Knowledge ready"
            } catch (t: Throwable) {
                isRagReady.value = false
                ragStatusText.value = "Index failed: ${t.message ?: t.javaClass.simpleName}"
            } finally {
                isIndexing.value = false
            }
        }
    }

    fun send(userText: String) {
        if (!canSendNow(userText)) return

        messages.add(ChatMessage(Role.User, userText))
        // placeholder model message (streaming updates will overwrite)
        messages.add(ChatMessage(Role.Model, ""))

        viewModelScope.launch(Dispatchers.Default) {
            try {
                pipeline.generateResponse(userText) { partial ->
                    viewModelScope.launch(Dispatchers.Main) {
                        val lastIdx = messages.lastIndex
                        if (lastIdx >= 0 && messages[lastIdx].role == Role.Model) {
                            messages[lastIdx] = ChatMessage(Role.Model, partial)
                        }
                    }
                }
            } catch (t: Throwable) {
                val errorText = "Error: ${t.message ?: t.javaClass.simpleName}"

                withContext(Dispatchers.Main) {
                    val lastIdx = messages.lastIndex
                    if (lastIdx >= 0 && messages[lastIdx].role == Role.Model) {
                        messages[lastIdx] = ChatMessage(Role.Model, errorText)
                    } else {
                        messages.add(ChatMessage(Role.Model, errorText))
                    }

                    val notInitialized = (t.message ?: "").lowercase().contains("not initialized")
                    if (notInitialized) {
                        isLlmReady.value = false
                        llmStatusText.value = "LLM not ready"
                    }
                }

                // Keep your existing retry-init behavior (not a reset-button fix)
                val notInitialized = (t.message ?: "").lowercase().contains("not initialized")
                if (notInitialized) {
                    startPipelineInit()
                }
            }
        }
    }

    fun newChat() {
        // Keep your current reset behavior unchanged (per your request).
        messages.clear()

        isIndexing.value = false
        isRagReady.value = false
        loadedFileName.value = null
        ragStatusText.value = "No .txt loaded"

        // Recreate pipeline (clears in-memory vector store)
        pipeline = RagPipeline(getApplication())

        // Reset per-model readiness
        startPipelineInit()
    }
}
