use crate::{
    doc_loader::Document,
    embeddings::{OPENAI_CLIENT, cosine_similarity},
    error::ServerError, // Keep ServerError for ::new()
};
use async_openai::{
    types::{
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs, CreateEmbeddingRequestArgs,
    },
    // Client as OpenAIClient, // Removed unused import
};
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
use serde::Deserialize; // Import Deserialize
use serde_json::json;
use std::{/* borrow::Cow, */ env, sync::Arc}; // Removed borrow::Cow
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
    crate_name: Arc<String>, // Use Arc for cheap cloning
    documents: Arc<Vec<Document>>,
    embeddings: Arc<Vec<(String, Array1<f32>)>>,
    peer: Arc<Mutex<Option<Peer<RoleServer>>>>, // Uses tokio::sync::Mutex
    startup_message: Arc<Mutex<Option<String>>>, // Keep the message itself
    startup_message_sent: Arc<Mutex<bool>>,     // Flag to track if sent (using tokio::sync::Mutex)
                                                // tool_name and info are handled by ServerHandler/macros now
}

impl RustDocsServer {
    // Updated constructor
    pub fn new(
        crate_name: String,
        documents: Vec<Document>,
        embeddings: Vec<(String, Array1<f32>)>,
        startup_message: String,
    ) -> Result<Self, ServerError> {
        // Keep ServerError for potential future init errors
        Ok(Self {
            crate_name: Arc::new(crate_name),
            documents: Arc::new(documents),
            embeddings: Arc::new(embeddings),
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

        let embedding_model: String =
            env::var("EMBEDDING_MODEL").unwrap_or_else(|_| "text-embedding-3-small".to_string());
        let question_embedding_request = CreateEmbeddingRequestArgs::default()
            .model(embedding_model)
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
            Some((best_path, _score)) => {
                eprintln!("Best match found: {}", best_path);
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

                    let chat_response = client.chat().create(chat_request).await.map_err(|e| {
                        McpError::internal_error(format!("OpenAI chat API error: {}", e), None)
                    })?;

                    chat_response
                        .choices
                        .first()
                        .and_then(|choice| choice.message.content.clone())
                        .unwrap_or_else(|| "Error: No response from LLM.".to_string())
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

    async fn call_tool(
        &self,
        request: rmcp::model::CallToolRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let dynamic_tool_name = format!("query_{}_docs", self.crate_name.replace('-', "_"));
        
        if request.name == dynamic_tool_name {
            // Parse the arguments for our tool
            let args: QueryRustDocsArgs = serde_json::from_value(request.arguments.into())
                .map_err(|e| McpError::invalid_params(format!("Invalid arguments: {}", e), None))?;
            
            // Call our existing tool method
            self.query_rust_docs(args).await
        } else {
            Err(McpError::invalid_params(
                format!("Tool '{}' not found", request.name),
                None,
            ))
        }
    }

    async fn list_tools(
        &self,
        _request: rmcp::model::PaginatedRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<rmcp::model::ListToolsResult, McpError> {
        let dynamic_tool_name = format!("query_{}_docs", self.crate_name.replace('-', "_"));
        
        let mut generator = schemars::r#gen::SchemaGenerator::default();
        let schema = QueryRustDocsArgs::json_schema(&mut generator);
        
        let tool = rmcp::model::Tool {
            name: dynamic_tool_name.into(),
            description: format!(
                "Query documentation for the '{}' crate using semantic search and LLM summarization.",
                self.crate_name
            ).into(),
            input_schema: serde_json::to_value(schema)
                .map_err(|e| McpError::internal_error(format!("Failed to generate schema: {}", e), None))?
                .as_object()
                .cloned()
                .map(Arc::new)
                .unwrap_or_else(|| Arc::new(serde_json::Map::new())),
        };

        Ok(rmcp::model::ListToolsResult {
            tools: vec![tool],
            next_cursor: None,
        })
    }
}
