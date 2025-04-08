use crate::{
    // doc_loader::Document, // Remove Document import
    embeddings::{OPENAI_CLIENT, cosine_similarity, CachedChunkEmbedding}, // Add CachedChunkEmbedding
    error::ServerError,
};
use async_openai::{
    types::{
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs, CreateEmbeddingRequestArgs,
    },
    // Client as OpenAIClient, // Removed unused import
};
use ndarray::ArrayView1; // Import ArrayView1
use ndarray::Array1;
use rmcp::model::AnnotateAble; // Import trait for .no_annotation()
use rmcp::{
    Error as McpError,
    Peer,
    ServerHandler, // Import necessary rmcp items
    model::{
        CallToolResult,
        Content,
        GetPromptRequestParam,
        GetPromptResult,
        /* EmptyObject, ErrorCode, */ Implementation,
        ListPromptsResult, // Removed EmptyObject, ErrorCode
        ListResourceTemplatesResult,
        ListResourcesResult,
        LoggingLevel, // Uncommented ListToolsResult
        LoggingMessageNotification,
        LoggingMessageNotificationMethod,
        LoggingMessageNotificationParam,
        Notification,
        PaginatedRequestParam,
        ProtocolVersion,
        RawResource,
        /* Prompt, PromptArgument, PromptMessage, PromptMessageContent, PromptMessageRole, */ // Removed Prompt types
        ReadResourceRequestParam,
        ReadResourceResult,
        Resource,
        ResourceContents,
        ServerCapabilities,
        ServerInfo,
        ServerNotification,
    },
    service::{RequestContext, RoleServer},
    tool,
};
use schemars::JsonSchema; // Import JsonSchema
use ordered_float::OrderedFloat; // For using f32 in BinaryHeap
use serde::Deserialize; // Import Deserialize
use serde_json::json;
use std::{
    cmp::Reverse, // For max-heap behavior with BinaryHeap
    collections::BinaryHeap, // For efficient Top-K retrieval
    env,
    sync::Arc,
};
use tokio::sync::Mutex;

// --- Argument Struct for the Tool ---

#[derive(Debug, Deserialize, JsonSchema)]
struct QueryRustDocsArgs {
    #[schemars(description = "The specific question about the crate's API or usage.")]
    question: String,
    // Removed crate_name field as it's implicit to the server instance
}

// --- Main Server Struct ---

// No longer needs ServerState, holds data directly
#[derive(Clone)] // Add Clone for tool macro requirements
pub struct RustDocsServer {
    crate_name: Arc<String>,
    // Store the combined chunk data directly
    cached_chunks: Arc<Vec<CachedChunkEmbedding>>,
    peer: Arc<Mutex<Option<Peer<RoleServer>>>>, // Uses tokio::sync::Mutex
    startup_message: Arc<Mutex<Option<String>>>, // Keep the message itself
    startup_message_sent: Arc<Mutex<bool>>,     // Flag to track if sent (using tokio::sync::Mutex)
}

impl RustDocsServer {
    /// Creates a new instance of the RustDocsServer.
    pub fn new(
        crate_name: String,
        // Accept the combined cached chunk data
        cached_chunks: Vec<CachedChunkEmbedding>,
        startup_message: String,
    ) -> Result<Self, ServerError> {
        // Keep ServerError for potential future init errors
        Ok(Self {
            crate_name: Arc::new(crate_name),
            // Store the Arc'd vector of cached chunks
            cached_chunks: Arc::new(cached_chunks),
            peer: Arc::new(Mutex::new(None)), // Uses tokio::sync::Mutex
            startup_message: Arc::new(Mutex::new(Some(startup_message))), // Initialize message
            startup_message_sent: Arc::new(Mutex::new(false)), // Initialize flag to false
        })
    }

    // Helper function to send log messages via MCP notification (remains mostly the same)
    pub fn send_log(&self, level: LoggingLevel, message: String) {
        let peer_arc = Arc::clone(&self.peer);
        tokio::spawn(async move {
            let mut peer_guard = peer_arc.lock().await;
            if let Some(peer) = peer_guard.as_mut() {
                let params = LoggingMessageNotificationParam {
                    level,
                    logger: None,
                    data: serde_json::Value::String(message),
                };
                let log_notification: LoggingMessageNotification = Notification {
                    method: LoggingMessageNotificationMethod,
                    params,
                };
                let server_notification =
                    ServerNotification::LoggingMessageNotification(log_notification);
                if let Err(e) = peer.send_notification(server_notification).await {
                    eprintln!("Failed to send MCP log notification: {}", e);
                }
            } else {
                eprintln!("Log task ran but MCP peer was not connected.");
            }
        });
    }

    // Helper for creating simple text resources (like in counter example)
    fn _create_resource_text(&self, uri: &str, name: &str) -> Resource {
        RawResource::new(uri, name.to_string()).no_annotation()
    }
}

// --- Tool Implementation ---

#[tool(tool_box)] // Add tool_box here as well, mirroring the example
// Tool methods go in a regular impl block
impl RustDocsServer {
    // Define the tool using the tool macro
    // Name removed; will be handled dynamically by overriding list_tools/get_tool
    #[tool(
        description = "Query documentation for a specific Rust crate using semantic search and LLM summarization."
    )]
    async fn query_rust_docs(
        &self,
        #[tool(aggr)] // Aggregate arguments into the struct
        args: QueryRustDocsArgs,
    ) -> Result<CallToolResult, McpError> {
        // --- Send Startup Message (if not already sent) ---
        let mut sent_guard = self.startup_message_sent.lock().await;
        if !*sent_guard {
            let mut msg_guard = self.startup_message.lock().await;
            if let Some(message) = msg_guard.take() {
                // Take the message out
                self.send_log(LoggingLevel::Info, message);
                *sent_guard = true; // Mark as sent
            }
            // Drop guards explicitly to avoid holding locks longer than needed
            drop(msg_guard);
            drop(sent_guard);
        } else {
            // Drop guard if already sent
            drop(sent_guard);
        }

        // Argument validation for crate_name removed

        let question = &args.question;

        // Log received query via MCP
        self.send_log(
            LoggingLevel::Info,
            format!(
                "Received query for crate '{}': {}",
                self.crate_name, question
            ),
        );

        // --- Embedding Generation for Question ---
        let client = OPENAI_CLIENT
            .get()
            .ok_or_else(|| McpError::internal_error("OpenAI client not initialized", None))?;

        let question_embedding_request = CreateEmbeddingRequestArgs::default()
            .model("text-embedding-3-small")
            .input(question.to_string())
            .build()
            .map_err(|e| {
                McpError::internal_error(format!("Failed to build embedding request: {}", e), None)
            })?;

        let question_embedding_response = client
            .embeddings()
            .create(question_embedding_request)
            .await
            .map_err(|e| McpError::internal_error(format!("OpenAI API error: {}", e), None))?;

        let question_embedding = question_embedding_response.data.first().ok_or_else(|| {
            McpError::internal_error("Failed to get embedding for question", None)
        })?;

        let question_vector = Array1::from(question_embedding.embedding.clone());

        // --- Find Top K Matching Chunks ---
        const K: usize = 5; // Number of top chunks to retrieve
        // Use a Min-Heap to keep track of the top K highest scores.
        // Store (Reverse<OrderedFloat<f32>>, usize) where usize is the index
        // into self.cached_chunks. This avoids needing Ord on CachedChunkEmbedding.
        let mut top_k_heap = BinaryHeap::with_capacity(K + 1);

        // Iterate through the cached chunks with their indices
        for (index, chunk) in self.cached_chunks.iter().enumerate() {
            // Calculate similarity between question and chunk's vector
            let chunk_vector_view = ArrayView1::from(&chunk.vector);
            let score = cosine_similarity(question_vector.view(), chunk_vector_view);

            // Wrap score for ordering and heap compatibility
            let ordered_score = OrderedFloat(score);

            // Push the score and the chunk's index onto the heap
            top_k_heap.push((Reverse(ordered_score), index));

            // If the heap exceeds K elements, remove the smallest score
            if top_k_heap.len() > K {
                top_k_heap.pop();
            }
        }

        // Extract the top K chunks from the heap. They will be in ascending score order.
        // Convert the heap into a sorted vector (descending score) for easier processing.
        // Extract the top K (score, index) pairs from the heap.
        // Convert the heap into a sorted vector (descending score).
        let top_k_indices_scores: Vec<_> = top_k_heap.into_sorted_vec();

        // --- Generate Response using LLM with Best Chunk Context ---
        // --- Generate Response using LLM with Top K Chunks Context ---
        // Use standard if/else statement instead of expression for assignment
        let response_text: String;
        if top_k_indices_scores.is_empty() {
             response_text = "Could not find any relevant document chunk context.".to_string();
        } else {
            // Combine the content of the top K chunks into a single context string
            let combined_context = top_k_indices_scores
                .iter()
                .rev() // Highest score first
                .map(|(Reverse(score), index)| { // Iterate over (score, index)
                    // Get the actual chunk using the index *before* formatting
                    let chunk = &self.cached_chunks[*index];
                    format!(
                        "Source: {} (Chunk {})\nScore: {:.4}\nContent:\n{}\n---\n",
                        chunk.source_path, chunk.chunk_index, score.into_inner(), chunk.content
                    ) // format! is now the return value of the closure
                })
                .collect::<String>();

             eprintln!(
                 "Found {} relevant chunks. Combined context length: {} chars.", // Use correct variable for count
                 top_k_indices_scores.len(), combined_context.len()
             );

             // Update prompts
             let system_prompt = format!( /* ... same as before ... */
                 "You are an expert technical assistant for the Rust crate '{}'. \
                  Answer the user's question based *only* on the provided context snippets below. \
                  Synthesize the information from all relevant snippets. \
                  If the context does not contain the answer, state that clearly. \
                  Do not make up information. Be clear, concise, and comprehensive, providing example usage code when possible.",
                 self.crate_name
             );
             let user_prompt = format!( /* ... same as before ... */
                 "Context Snippets:\n===\n{}\n===\n\nQuestion: {}",
                 combined_context, question
             );

            // Perform the chat completion request, handling errors internally to produce a String
            let system_message_result = ChatCompletionRequestSystemMessageArgs::default()
                .content(system_prompt)
                .build();
            let user_message_result = ChatCompletionRequestUserMessageArgs::default()
                .content(user_prompt)
                .build();

            if let Err(e) = system_message_result {
                response_text = format!("Error building system message: {}", e);
            } else if let Err(e) = user_message_result {
                response_text = format!("Error building user message: {}", e);
            } else {
                // Messages built successfully, proceed to build request
                let chat_request_result = CreateChatCompletionRequestArgs::default()
                    .model("gpt-4o-mini-2024-07-18")
                    .messages(vec![system_message_result.unwrap().into(), user_message_result.unwrap().into()])
                    .build();

                if let Err(e) = chat_request_result {
                     response_text = format!("Error building chat request: {}", e);
                } else {
                     // Request built successfully, make the API call
                     match client.chat().create(chat_request_result.unwrap()).await {
                         Ok(chat_response) => {
                             // Extract the response content or provide a default error message
                             response_text = chat_response
                                 .choices
                                 .first()
                                 .and_then(|choice| choice.message.content.clone())
                                 .unwrap_or_else(|| "Error: No response content from LLM.".to_string());
                         }
                         Err(e) => {
                             response_text = format!("OpenAI chat API error: {}", e); // Assign error string on API failure
                         }
                     }
                }
            }
        } // End of else block

        // --- Format and Return Result ---
        Ok(CallToolResult::success(vec![Content::text(format!(
            "From {} docs: {}",
            self.crate_name, response_text
        ))]))
    }
}

// --- ServerHandler Implementation ---

#[tool(tool_box)] // Use imported tool macro directly
impl ServerHandler for RustDocsServer {
    fn get_info(&self) -> ServerInfo {
        // Define capabilities using the builder
        let capabilities = ServerCapabilities::builder()
            .enable_tools() // Enable tools capability
            .enable_logging() // Enable logging capability
            // Add other capabilities like resources, prompts if needed later
            .build();

        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05, // Use latest known version
            capabilities,
            server_info: Implementation {
                name: "rust-docs-mcp-server".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            // Provide instructions based on the specific crate
            instructions: Some(format!(
                "This server provides tools to query documentation for the '{}' crate. \
                 Use the 'query_rust_docs' tool with a specific question to get information \
                 about its API, usage, and examples, derived from its official documentation.",
                self.crate_name
            )),
        }
    }

    // --- Placeholder Implementations for other ServerHandler methods ---
    // Implement these properly if resource/prompt features are added later.

    async fn list_resources(
        &self,
        _request: PaginatedRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        // Example: Return the crate name as a resource
        Ok(ListResourcesResult {
            resources: vec![
                self._create_resource_text(&format!("crate://{}", self.crate_name), "crate_name"),
            ],
            next_cursor: None,
        })
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, McpError> {
        let expected_uri = format!("crate://{}", self.crate_name);
        if request.uri == expected_uri {
            Ok(ReadResourceResult {
                contents: vec![ResourceContents::text(
                    self.crate_name.as_str(), // Explicitly get &str from Arc<String>
                    &request.uri,
                )],
            })
        } else {
            Err(McpError::resource_not_found(
                format!("Resource URI not found: {}", request.uri),
                Some(json!({ "uri": request.uri })),
            ))
        }
    }

    async fn list_prompts(
        &self,
        _request: PaginatedRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListPromptsResult, McpError> {
        Ok(ListPromptsResult {
            next_cursor: None,
            prompts: Vec::new(), // No prompts defined yet
        })
    }

    async fn get_prompt(
        &self,
        request: GetPromptRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<GetPromptResult, McpError> {
        Err(McpError::invalid_params(
            // Or prompt_not_found if that exists
            format!("Prompt not found: {}", request.name),
            None,
        ))
    }

    async fn list_resource_templates(
        &self,
        _request: PaginatedRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, McpError> {
        Ok(ListResourceTemplatesResult {
            next_cursor: None,
            resource_templates: Vec::new(), // No templates defined yet
        })
    }

}
