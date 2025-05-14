use crate::{
    doc_loader::Document,
    embeddings::{OPENAI_CLIENT, OLLAMA_CLIENT, cosine_similarity, generate_single_ollama_embedding},
    error::ServerError,
};
use async_openai::{
    types::{
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs, CreateEmbeddingRequestArgs,
    },
};
use ndarray::Array1;
use rmcp::model::AnnotateAble;
use rmcp::{
    Error as McpError,
    Peer,
    ServerHandler,
    model::{
        CallToolResult,
        Content,
        GetPromptRequestParam,
        GetPromptResult,
        Implementation,
        ListPromptsResult,
        ListResourceTemplatesResult,
        ListResourcesResult,
        LoggingLevel,
        LoggingMessageNotification,
        LoggingMessageNotificationMethod,
        LoggingMessageNotificationParam,
        Notification,
        PaginatedRequestParam,
        ProtocolVersion,
        RawResource,
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
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use std::{env, sync::Arc};
use tokio::sync::Mutex;

// Add Ollama imports for chat completion - using the correct import paths
use ollama_rs::generation::chat::{ChatMessage, MessageRole};
use ollama_rs::generation::chat::request::ChatMessageRequest;

// --- Argument Struct for the Tool ---

#[derive(Debug, Deserialize, JsonSchema)]
struct QueryRustDocsArgs {
    #[schemars(description = "The specific question about the crate's API or usage.")]
    question: String,
}

// --- Main Server Struct ---

#[derive(Clone)]
pub struct RustDocsServer {
    crate_name: Arc<String>,
    documents: Arc<Vec<Document>>,
    embeddings: Arc<Vec<(String, Array1<f32>)>>,
    peer: Arc<Mutex<Option<Peer<RoleServer>>>>,
    startup_message: Arc<Mutex<Option<String>>>,
    startup_message_sent: Arc<Mutex<bool>>,
}

impl RustDocsServer {
    pub fn new(
        crate_name: String,
        documents: Vec<Document>,
        embeddings: Vec<(String, Array1<f32>)>,
        startup_message: String,
    ) -> Result<Self, ServerError> {
        Ok(Self {
            crate_name: Arc::new(crate_name),
            documents: Arc::new(documents),
            embeddings: Arc::new(embeddings),
            peer: Arc::new(Mutex::new(None)),
            startup_message: Arc::new(Mutex::new(Some(startup_message))),
            startup_message_sent: Arc::new(Mutex::new(false)),
        })
    }

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

    fn _create_resource_text(&self, uri: &str, name: &str) -> Resource {
        RawResource::new(uri, name.to_string()).no_annotation()
    }

    /// Generate embedding for a question using the same model that was used for documents
    async fn generate_question_embedding(&self, question: &str) -> Result<Array1<f32>, McpError> {
        // First try Ollama (preferred for consistency)
        if let Some(ollama_client) = OLLAMA_CLIENT.get() {
            match generate_single_ollama_embedding(ollama_client, question, "nomic-embed-text").await {
                Ok(embedding) => return Ok(embedding),
                Err(e) => {
                    eprintln!("Failed to generate question embedding with Ollama: {}", e);
                    self.send_log(
                        LoggingLevel::Warning,
                        format!("Ollama embedding failed, trying OpenAI fallback: {}", e),
                    );
                }
            }
        }

        // Fallback to OpenAI if Ollama fails or is not available
        if let Some(openai_client) = OPENAI_CLIENT.get() {
            let embedding_model: String = env::var("EMBEDDING_MODEL")
                .unwrap_or_else(|_| "text-embedding-3-small".to_string());
            
            let question_embedding_request = CreateEmbeddingRequestArgs::default()
                .model(embedding_model)
                .input(question.to_string())
                .build()
                .map_err(|e| {
                    McpError::internal_error(format!("Failed to build embedding request: {}", e), None)
                })?;

            let question_embedding_response = openai_client
                .embeddings()
                .create(question_embedding_request)
                .await
                .map_err(|e| McpError::internal_error(format!("OpenAI API error: {}", e), None))?;

            let question_embedding = question_embedding_response.data.first().ok_or_else(|| {
                McpError::internal_error("Failed to get embedding for question", None)
            })?;

            return Ok(Array1::from(question_embedding.embedding.clone()));
        }

        Err(McpError::internal_error(
            "No embedding client available (neither Ollama nor OpenAI)",
            None,
        ))
    }

    /// Generate chat completion using Ollama or OpenAI
    async fn generate_chat_completion(
        &self,
        system_prompt: &str,
        user_prompt: &str,
    ) -> Result<String, McpError> {
        // First try Ollama (preferred for consistency)
        if let Some(ollama_client) = OLLAMA_CLIENT.get() {
            // Get the chat model from environment variable, default to llama3.2
            let chat_model = env::var("OLLAMA_CHAT_MODEL")
                .unwrap_or_else(|_| "llama3.2".to_string());
            
            self.send_log(
                LoggingLevel::Info,
                format!("Using Ollama for chat completion with model: {}", chat_model),
            );

            // Create the chat messages - system message followed by user message
            let messages = vec![
                ChatMessage::system(system_prompt.to_string()),
                ChatMessage::user(user_prompt.to_string()),
            ];

            // Create the chat request
            let chat_request = ChatMessageRequest::new(chat_model, messages);

            match ollama_client.send_chat_messages(chat_request).await {
                Ok(response) => {
                    // The response.message is a ChatMessage directly, not an Option
                    // We need to access its content field
                    return Ok(response.message.content);
                }
                Err(e) => {
                    eprintln!("Failed to generate chat completion with Ollama: {}", e);
                    self.send_log(
                        LoggingLevel::Warning,
                        format!("Ollama chat failed, trying OpenAI fallback: {}", e),
                    );
                }
            }
        }

        // Fallback to OpenAI if Ollama fails or is not available
        if let Some(openai_client) = OPENAI_CLIENT.get() {
            self.send_log(
                LoggingLevel::Info,
                "Using OpenAI for chat completion".to_string(),
            );

            let llm_model: String = env::var("LLM_MODEL")
                .unwrap_or_else(|_| "gpt-4o-mini-2024-07-18".to_string());

            let chat_request = CreateChatCompletionRequestArgs::default()
                .model(llm_model)
                .messages(vec![
                    ChatCompletionRequestSystemMessageArgs::default()
                        .content(system_prompt)
                        .build()
                        .map_err(|e| {
                            McpError::internal_error(
                                format!("Failed to build system message: {}", e),
                                None,
                            )
                        })?
                        .into(),
                    ChatCompletionRequestUserMessageArgs::default()
                        .content(user_prompt)
                        .build()
                        .map_err(|e| {
                            McpError::internal_error(
                                format!("Failed to build user message: {}", e),
                                None,
                            )
                        })?
                        .into(),
                ])
                .build()
                .map_err(|e| {
                    McpError::internal_error(
                        format!("Failed to build chat request: {}", e),
                        None,
                    )
                })?;

            let chat_response = openai_client.chat().create(chat_request).await.map_err(|e| {
                McpError::internal_error(format!("OpenAI chat API error: {}", e), None)
            })?;

            return chat_response
                .choices
                .first()
                .and_then(|choice| choice.message.content.clone())
                .ok_or_else(|| McpError::internal_error("No response from OpenAI", None));
        }

        Err(McpError::internal_error(
            "No chat client available (neither Ollama nor OpenAI)",
            None,
        ))
    }
}

// --- Tool Implementation ---

#[tool(tool_box)]
impl RustDocsServer {
    #[tool(
        description = "Query documentation for a specific Rust crate using semantic search and LLM summarization."
    )]
    async fn query_rust_docs(
        &self,
        #[tool(aggr)]
        args: QueryRustDocsArgs,
    ) -> Result<CallToolResult, McpError> {
        // --- Send Startup Message (if not already sent) ---
        let mut sent_guard = self.startup_message_sent.lock().await;
        if !*sent_guard {
            let mut msg_guard = self.startup_message.lock().await;
            if let Some(message) = msg_guard.take() {
                self.send_log(LoggingLevel::Info, message);
                *sent_guard = true;
            }
            drop(msg_guard);
            drop(sent_guard);
        } else {
            drop(sent_guard);
        }

        let question = &args.question;

        // Log received query via MCP
        self.send_log(
            LoggingLevel::Info,
            format!(
                "Received query for crate '{}': {}",
                self.crate_name, question
            ),
        );

        // --- Generate question embedding using the same model as documents ---
        let question_vector = self.generate_question_embedding(question).await?;

        // --- Find Best Matching Document ---
        let mut best_match: Option<(&str, f32)> = None;
        for (path, doc_embedding) in self.embeddings.iter() {
            let score = cosine_similarity(question_vector.view(), doc_embedding.view());
            if best_match.is_none() || score > best_match.unwrap().1 {
                best_match = Some((path, score));
            }
        }

        // --- Generate Response using LLM ---
        let response_text = match best_match {
            Some((best_path, score)) => {
                eprintln!("Best match found: {} (similarity: {:.3})", best_path, score);
                
                // Log the similarity score via MCP
                self.send_log(
                    LoggingLevel::Info,
                    format!("Best matching document: {} (similarity: {:.3})", best_path, score),
                );

                let context_doc = self.documents.iter().find(|doc| doc.path == best_path);

                if let Some(doc) = context_doc {
                    let system_prompt = format!(
                        "You are an expert technical assistant for the Rust crate '{}'. \
                         Answer the user's question based *only* on the provided context. \
                         If the context does not contain the answer, say so. \
                         Do not make up information. Be clear, concise, and comprehensive providing example usage code when possible.",
                        self.crate_name
                    );
                    let user_prompt = format!(
                        "Context:\n---\n{}\n---\n\nQuestion: {}",
                        doc.content, question
                    );

                    // Use the new chat completion method that supports both Ollama and OpenAI
                    self.generate_chat_completion(&system_prompt, &user_prompt).await?
                } else {
                    "Error: Could not find content for best matching document.".to_string()
                }
            }
            None => "Could not find any relevant document context.".to_string(),
        };

        // --- Format and Return Result ---
        Ok(CallToolResult::success(vec![Content::text(format!(
            "From {} docs: {}",
            self.crate_name, response_text
        ))]))
    }
}

// --- ServerHandler Implementation ---

#[tool(tool_box)]
impl ServerHandler for RustDocsServer {
    fn get_info(&self) -> ServerInfo {
        let capabilities = ServerCapabilities::builder()
            .enable_tools()
            .enable_logging()
            .build();

        // Determine which embedding and chat models are being used
        let model_info = if OLLAMA_CLIENT.get().is_some() {
            let chat_model = env::var("OLLAMA_CHAT_MODEL")
                .unwrap_or_else(|_| "llama3.2".to_string());
            format!("locally with Ollama (nomic-embed-text for embeddings, {} for chat)", chat_model)
        } else {
            "with OpenAI".to_string()
        };

        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities,
            server_info: Implementation {
                name: "rust-docs-mcp-server".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            instructions: Some(format!(
                "This server provides tools to query documentation for the '{}' crate. \
                 Use the 'query_rust_docs' tool with a specific question to get information \
                 about its API, usage, and examples, derived from its official documentation. \
                 Running {}.",
                self.crate_name, model_info
            )),
        }
    }

    async fn list_resources(
        &self,
        _request: PaginatedRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
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
                    self.crate_name.as_str(),
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
            prompts: Vec::new(),
        })
    }

    async fn get_prompt(
        &self,
        request: GetPromptRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<GetPromptResult, McpError> {
        Err(McpError::invalid_params(
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
            resource_templates: Vec::new(),
        })
    }
}