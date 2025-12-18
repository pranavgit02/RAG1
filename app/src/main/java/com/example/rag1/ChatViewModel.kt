package com.example.rag1

import android.app.Application
import android.net.Uri
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

enum class Role { User, Model }

data class ChatMessage(
    val role: Role,
    val text: String,
)

class ChatViewModel(app: Application) : AndroidViewModel(app) {

    private var pipeline = RagPipeline(app)

    val messages = mutableStateListOf<ChatMessage>()

    // Top status
    val modelStatusText = mutableStateOf("initializing model...")
    val isModelReady = mutableStateOf(false)

    val ragStatusText = mutableStateOf("No .txt loaded")
    val isIndexing = mutableStateOf(false)
    val isRagReady = mutableStateOf(false)
    val loadedFileName = mutableStateOf<String?>(null)

    init {
        // kick off model init
        viewModelScope.launch {
            try {
                pipeline.awaitModelReady()
                isModelReady.value = true
                modelStatusText.value = "Model ready"
            } catch (t: Throwable) {
                isModelReady.value = false
                modelStatusText.value = "Model init failed"
                ragStatusText.value = "Error: ${t.message ?: t.javaClass.simpleName}"
            }
        }
    }

    fun canSendNow(currentInput: String): Boolean {
        if (currentInput.isBlank()) return false
        return isModelReady.value && isRagReady.value && !isIndexing.value
    }

    fun onPickTxt(uri: Uri, displayName: String?) {
        val ctx = getApplication<Application>().applicationContext
        viewModelScope.launch {
            isIndexing.value = true
            isRagReady.value = false
            loadedFileName.value = displayName ?: "selected_file.txt"
            ragStatusText.value = "Reading file..."

            try {
                val rawText = withContext(Dispatchers.IO) {
                    ctx.contentResolver.openInputStream(uri)?.use { it.readBytes().toString(Charsets.UTF_8) }
                        ?: throw IllegalStateException("Could not open file")
                }

                ragStatusText.value = "Chunking + indexing..."
                pipeline.indexUserText(rawText) { chunkCount ->
                    ragStatusText.value = "Indexing... ($chunkCount chunks)"
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

        viewModelScope.launch {
            try {
                pipeline.generateResponse(userText) { partial ->
                    // update last message
                    val lastIdx = messages.lastIndex
                    if (lastIdx >= 0 && messages[lastIdx].role == Role.Model) {
                        messages[lastIdx] = ChatMessage(Role.Model, partial)
                    }
                }
            } catch (t: Throwable) {
                val lastIdx = messages.lastIndex
                if (lastIdx >= 0 && messages[lastIdx].role == Role.Model) {
                    messages[lastIdx] = ChatMessage(Role.Model, "Error: ${t.message ?: t.javaClass.simpleName}")
                } else {
                    messages.add(ChatMessage(Role.Model, "Error: ${t.message ?: t.javaClass.simpleName}"))
                }
            }
        }
    }

    fun newChat() {
        // wipe UI + wipe vector DB (in-memory) by recreating pipeline
        messages.clear()

        isIndexing.value = false
        isRagReady.value = false
        loadedFileName.value = null
        ragStatusText.value = "No .txt loaded"

        isModelReady.value = false
        modelStatusText.value = "initializing model..."

        pipeline = RagPipeline(getApplication())

        viewModelScope.launch {
            try {
                pipeline.awaitModelReady()
                isModelReady.value = true
                modelStatusText.value = "Model ready"
            } catch (t: Throwable) {
                isModelReady.value = false
                modelStatusText.value = "Model init failed"
                ragStatusText.value = "Error: ${t.message ?: t.javaClass.simpleName}"
            }
        }
    }
}
