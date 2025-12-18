package com.example.rag1

import android.database.Cursor
import android.net.Uri
import android.provider.OpenableColumns
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.AttachFile
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(vm: ChatViewModel) {
    val ctx = LocalContext.current
    val listState = rememberLazyListState()
    val scope = rememberCoroutineScope()

    var input by remember { mutableStateOf("") }

    val pickTxtLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        if (uri != null) {
            val name = queryDisplayName(ctx.contentResolver.query(uri, null, null, null, null))
            vm.onPickTxt(uri, name)
        }
    }

    LaunchedEffect(vm.messages.size) {
        if (vm.messages.isNotEmpty()) {
            listState.animateScrollToItem(vm.messages.size - 1)
        }
    }

    Scaffold(
        contentWindowInsets = WindowInsets.safeDrawing,
        topBar = {
            TopAppBar(
                title = { Text("Local RAG Chat") },
                actions = {
                    IconButton(onClick = { vm.newChat() }) {
                        Icon(Icons.Filled.Refresh, contentDescription = "New chat")
                    }
                    IconButton(onClick = { pickTxtLauncher.launch(arrayOf("text/plain", "text/*")) }) {
                        Icon(Icons.Filled.AttachFile, contentDescription = "Pick .txt")
                    }
                }
            )
        }
    ) { pad ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(pad)
                .padding(horizontal = 12.dp)
        ) {
            // Status line(s)
            Row(
                modifier = Modifier.fillMaxWidth().padding(top = 6.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    vm.modelStatusText.value,
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.weight(1f)
                )
                Text(
                    vm.ragStatusText.value,
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    textAlign = TextAlign.End,
                    modifier = Modifier.weight(1f)
                )
            }

            vm.loadedFileName.value?.let { fn ->
                Text(
                    text = "File: $fn",
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 2.dp, bottom = 6.dp)
                )
            }

            if (vm.isIndexing.value) {
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth().padding(bottom = 6.dp))
            }

            // Messages
            LazyColumn(
                modifier = Modifier.weight(1f).fillMaxWidth(),
                state = listState,
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(vm.messages) { msg ->
                    MessageBubble(msg)
                }
            }

            // Input
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .imePadding()
                    .navigationBarsPadding()
                    .padding(vertical = 10.dp),
                verticalAlignment = Alignment.Bottom
            ) {
                OutlinedTextField(
                    value = input,
                    onValueChange = { input = it },
                    modifier = Modifier.weight(1f),
                    label = { Text("Message") },
                    enabled = true,
                    supportingText = {
                        val canSend = vm.canSendNow(input)
                        if (!vm.isModelReady.value) Text("Wait for model initialization")
                        else if (!vm.isRagReady.value) Text("Pick a .txt and wait for indexing")
                        else if (vm.isIndexing.value) Text("Indexing in progress…")
                        else if (!canSend) Text("Type something")
                    }
                )
                Spacer(Modifier.width(8.dp))
                IconButton(
                    onClick = {
                        val text = input
                        input = ""
                        vm.send(text)
                        scope.launch { if (vm.messages.isNotEmpty()) listState.animateScrollToItem(vm.messages.size - 1) }
                    },
                    enabled = vm.canSendNow(input)
                ) {
                    Icon(Icons.AutoMirrored.Filled.Send, contentDescription = "Send")
                }
            }
        }
    }
}

@Composable
private fun MessageBubble(msg: ChatMessage) {
    val fromModel = msg.role == Role.Model
    val bg = if (fromModel) MaterialTheme.colorScheme.secondaryContainer else MaterialTheme.colorScheme.primaryContainer
    val fg = if (fromModel) MaterialTheme.colorScheme.onSecondaryContainer else MaterialTheme.colorScheme.onPrimaryContainer
    val align = if (fromModel) Alignment.CenterStart else Alignment.CenterEnd

    Box(modifier = Modifier.fillMaxWidth(), contentAlignment = align) {
        Text(
            text = msg.text,
            color = fg,
            modifier = Modifier
                .widthIn(max = 340.dp)
                .background(bg, RoundedCornerShape(14.dp))
                .padding(12.dp)
        )
    }
}

private fun queryDisplayName(cursor: Cursor?): String? {
    cursor ?: return null
    return cursor.use {
        val idx = it.getColumnIndex(OpenableColumns.DISPLAY_NAME)
        if (idx >= 0 && it.moveToFirst()) it.getString(idx) else null
    }
}
