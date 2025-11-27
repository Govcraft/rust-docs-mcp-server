use crate::{
    doc_loader::{self, Document},
    embeddings::{self, OPENAI_CLIENT, cosine_similarity, CachedDocumentEmbedding},
    error::ServerError, // Keep ServerError for ::new()
};
use bincode::config;
use cargo::core::PackageIdSpec;
use std::{
    collections::{HashMap, hash_map::DefaultHasher},
    fs::{self, File},
    hash::{Hash, Hasher},
    io::BufReader,
    path::PathBuf,
};
#[cfg(not(target_os = "windows"))]
use xdg::BaseDirectories;
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

// --- Argument Structs for Tools ---

// For single-crate mode (no package_spec needed)
#[derive(Debug, Deserialize, JsonSchema)]
struct QueryRustDocsArgs {
    #[schemars(description = "The specific question about the crate's API or usage.")]
    question: String,
}

// For any-crate mode (needs package_spec)
#[derive(Debug, Deserialize, JsonSchema)]
struct QueryAnyCrateDocsArgs {
    #[schemars(description = "The package ID specification (e.g., 'serde@^1.0', 'tokio').")]
    package_spec: String,
    #[schemars(description = "The specific question about the crate's API or usage.")]
    question: String,
    #[schemars(description = "Optional features to enable for the crate when generating documentation.")]
    features: Option<Vec<String>>,
}

// --- Main Server Struct for Single Crate Mode ---

#[derive(Clone)] // Add Clone for tool macro requirements
pub struct RustDocsSingleCrateServer {
    crate_name: Arc<String>, // Use Arc for cheap cloning
    documents: Arc<Vec<Document>>,
    embeddings: Arc<Vec<(String, Array1<f32>)>>,
    peer: Arc<Mutex<Option<Peer<RoleServer>>>>, // Uses tokio::sync::Mutex
    startup_message: Arc<Mutex<Option<String>>>, // Keep the message itself
    startup_message_sent: Arc<Mutex<bool>>,     // Flag to track if sent (using tokio::sync::Mutex)
}

impl RustDocsSingleCrateServer {
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

// --- Tool Implementation for Single Crate Mode ---

#[tool(tool_box)]
impl RustDocsSingleCrateServer {
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

// --- ServerHandler Implementation for Single Crate Mode ---

#[tool(tool_box)]
impl ServerHandler for RustDocsSingleCrateServer {
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

// --- Any-Crate Mode Server ---

#[derive(Clone)]
pub struct RustDocsAnyCrateServer {
    // Cache for loaded crate documentation
    cache: Arc<Mutex<HashMap<String, CrateCache>>>,
    peer: Arc<Mutex<Option<Peer<RoleServer>>>>,
}

struct CrateCache {
    crate_name: String,
    documents: Vec<Document>,
    embeddings: Vec<(String, Array1<f32>)>,
}

impl RustDocsAnyCrateServer {
    pub fn new() -> Result<Self, ServerError> {
        Ok(Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            peer: Arc::new(Mutex::new(None)),
        })
    }

    // Helper function to send log messages
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

    // Helper to hash features
    fn hash_features(features: &Option<Vec<String>>) -> String {
        features
            .as_ref()
            .map(|f| {
                let mut sorted_features = f.clone();
                sorted_features.sort_unstable();
                let mut hasher = DefaultHasher::new();
                sorted_features.hash(&mut hasher);
                format!("{:x}", hasher.finish())
            })
            .unwrap_or_else(|| "no_features".to_string())
    }

    // Helper to load or get cached crate documentation
    async fn get_or_load_crate(
        &self,
        package_spec: &str,
        features: Option<Vec<String>>,
    ) -> Result<(String, Vec<Document>, Vec<(String, Array1<f32>)>), McpError> {
        // Parse the package spec
        let spec = PackageIdSpec::parse(package_spec).map_err(|e| {
            McpError::invalid_params(
                format!("Failed to parse package ID spec '{}': {}", package_spec, e),
                None,
            )
        })?;

        let crate_name = spec.name().to_string();
        let crate_version_req = spec
            .version()
            .map(|v| v.to_string())
            .unwrap_or_else(|| "*".to_string());

        // Create cache key
        let features_hash = Self::hash_features(&features);
        let cache_key = format!("{}_{}_{}", crate_name, crate_version_req, features_hash);

        // Check if already in cache
        {
            let cache_guard = self.cache.lock().await;
            if let Some(cached) = cache_guard.get(&cache_key) {
                self.send_log(
                    LoggingLevel::Info,
                    format!("Using cached documentation for crate '{}'", crate_name),
                );
                return Ok((
                    cached.crate_name.clone(),
                    cached.documents.clone(),
                    cached.embeddings.clone(),
                ));
            }
        }

        self.send_log(
            LoggingLevel::Info,
            format!("Loading documentation for crate '{}' (version: {}, features: {:?})",
                crate_name, crate_version_req, features),
        );

        // Determine cache file path
        let sanitized_version_req = crate_version_req
            .replace(|c: char| !c.is_alphanumeric() && c != '.' && c != '-', "_");

        let embeddings_relative_path = PathBuf::from(&crate_name)
            .join(&sanitized_version_req)
            .join(&features_hash)
            .join("embeddings.bin");

        #[cfg(not(target_os = "windows"))]
        let embeddings_file_path = {
            let xdg_dirs = BaseDirectories::with_prefix("rustdocs-mcp-server")
                .map_err(|e| McpError::internal_error(format!("Failed to get XDG directories: {}", e), None))?;
            xdg_dirs
                .place_data_file(embeddings_relative_path)
                .map_err(|e| McpError::internal_error(format!("IO error: {}", e), None))?
        };

        #[cfg(target_os = "windows")]
        let embeddings_file_path = {
            let cache_dir = dirs::cache_dir().ok_or_else(|| {
                McpError::internal_error("Could not determine cache directory on Windows", None)
            })?;
            let app_cache_dir = cache_dir.join("rustdocs-mcp-server");
            fs::create_dir_all(&app_cache_dir)
                .map_err(|e| McpError::internal_error(format!("IO error: {}", e), None))?;
            app_cache_dir.join(embeddings_relative_path)
        };

        // Try to load from disk cache
        let mut loaded_embeddings: Option<Vec<(String, Array1<f32>)>> = None;
        let mut loaded_documents: Option<Vec<Document>> = None;

        if embeddings_file_path.exists() {
            match File::open(&embeddings_file_path) {
                Ok(file) => {
                    let reader = BufReader::new(file);
                    match bincode::decode_from_reader::<Vec<CachedDocumentEmbedding>, _, _>(
                        reader,
                        config::standard(),
                    ) {
                        Ok(cached_data) => {
                            let count = cached_data.len();
                            let mut embeddings = Vec::with_capacity(count);
                            let mut documents = Vec::with_capacity(count);
                            for item in cached_data {
                                embeddings.push((item.path.clone(), Array1::from(item.vector)));
                                documents.push(Document {
                                    path: item.path,
                                    content: item.content,
                                });
                            }
                            loaded_embeddings = Some(embeddings);
                            loaded_documents = Some(documents);
                            self.send_log(
                                LoggingLevel::Info,
                                format!("Loaded {} cached embeddings for crate '{}'", count, crate_name),
                            );
                        }
                        Err(e) => {
                            eprintln!("Failed to decode cache file: {}. Will regenerate.", e);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Failed to open cache file: {}. Will regenerate.", e);
                }
            }
        }

        // Generate embeddings if not cached
        let (final_documents, final_embeddings) = if let (Some(docs), Some(embeds)) = (loaded_documents, loaded_embeddings) {
            (docs, embeds)
        } else {
            // Load documents
            let documents = doc_loader::load_documents(&crate_name, &crate_version_req, features.as_ref())
                .map_err(|e| McpError::internal_error(format!("Failed to load documents: {}", e), None))?;

            // Generate embeddings
            let openai_client = OPENAI_CLIENT
                .get()
                .ok_or_else(|| McpError::internal_error("OpenAI client not initialized", None))?;

            let embedding_model: String = env::var("EMBEDDING_MODEL")
                .unwrap_or_else(|_| "text-embedding-3-small".to_string());

            let (embeddings, _total_tokens) = embeddings::generate_embeddings(openai_client, &documents, &embedding_model)
                .await
                .map_err(|e| McpError::internal_error(format!("Failed to generate embeddings: {}", e), None))?;

            // Save to disk cache
            let mut combined_cache_data: Vec<CachedDocumentEmbedding> = Vec::new();
            let embedding_map: HashMap<String, Array1<f32>> = embeddings.clone().into_iter().collect();

            for doc in &documents {
                if let Some(embedding_array) = embedding_map.get(&doc.path) {
                    combined_cache_data.push(CachedDocumentEmbedding {
                        path: doc.path.clone(),
                        content: doc.content.clone(),
                        vector: embedding_array.to_vec(),
                    });
                }
            }

            match bincode::encode_to_vec(&combined_cache_data, config::standard()) {
                Ok(encoded_bytes) => {
                    if let Some(parent_dir) = embeddings_file_path.parent() {
                        if !parent_dir.exists() {
                            let _ = fs::create_dir_all(parent_dir);
                        }
                    }
                    if let Err(e) = fs::write(&embeddings_file_path, encoded_bytes) {
                        eprintln!("Warning: Failed to write cache file: {}", e);
                    }
                }
                Err(e) => {
                    eprintln!("Warning: Failed to encode data for cache: {}", e);
                }
            }

            (documents, embeddings)
        };

        // Store in memory cache
        {
            let mut cache_guard = self.cache.lock().await;
            cache_guard.insert(
                cache_key,
                CrateCache {
                    crate_name: crate_name.clone(),
                    documents: final_documents.clone(),
                    embeddings: final_embeddings.clone(),
                },
            );
        }

        Ok((crate_name, final_documents, final_embeddings))
    }
}

// --- Tool Implementation for Any-Crate Mode ---

#[tool(tool_box)]
impl RustDocsAnyCrateServer {
    #[tool(
        description = "Query documentation for any Rust crate using semantic search and LLM summarization."
    )]
    async fn query_any_crate_docs(
        &self,
        #[tool(aggr)]
        args: QueryAnyCrateDocsArgs,
    ) -> Result<CallToolResult, McpError> {
        let (crate_name, documents, embeddings) = self
            .get_or_load_crate(&args.package_spec, args.features)
            .await?;

        self.send_log(
            LoggingLevel::Info,
            format!("Received query for crate '{}': {}", crate_name, args.question),
        );

        // --- Embedding Generation for Question ---
        let client = OPENAI_CLIENT
            .get()
            .ok_or_else(|| McpError::internal_error("OpenAI client not initialized", None))?;

        let embedding_model: String =
            env::var("EMBEDDING_MODEL").unwrap_or_else(|_| "text-embedding-3-small".to_string());

        let question_embedding_request = CreateEmbeddingRequestArgs::default()
            .model(embedding_model)
            .input(args.question.to_string())
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
        for (path, doc_embedding) in embeddings.iter() {
            let score = cosine_similarity(question_vector.view(), doc_embedding.view());
            if best_match.is_none() || score > best_match.unwrap().1 {
                best_match = Some((path, score));
            }
        }

        // --- Generate Response using LLM ---
        let response_text = match best_match {
            Some((best_path, _score)) => {
                eprintln!("Best match found: {}", best_path);
                let context_doc = documents.iter().find(|doc| doc.path == best_path);

                if let Some(doc) = context_doc {
                    let system_prompt = format!(
                        "You are an expert technical assistant for the Rust crate '{}'. \
                         Answer the user's question based *only* on the provided context. \
                         If the context does not contain the answer, say so. \
                         Do not make up information. Be clear, concise, and comprehensive providing example usage code when possible.",
                        crate_name
                    );
                    let user_prompt = format!(
                        "Context:\n---\n{}\n---\n\nQuestion: {}",
                        doc.content, args.question
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
            crate_name, response_text
        ))]))
    }
}

// --- ServerHandler Implementation for Any-Crate Mode ---

#[tool(tool_box)]
impl ServerHandler for RustDocsAnyCrateServer {
    fn get_info(&self) -> ServerInfo {
        let capabilities = ServerCapabilities::builder()
            .enable_tools()
            .enable_logging()
            .build();

        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities,
            server_info: Implementation {
                name: "rust-docs-mcp-server".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            instructions: Some(
                "This server provides tools to query documentation for any Rust crate. \
                 Use the 'query_any_crate_docs' tool with a package specification and question \
                 to get information about any crate's API, usage, and examples."
                    .to_string(),
            ),
        }
    }

    async fn list_resources(
        &self,
        _request: PaginatedRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        Ok(ListResourcesResult {
            resources: vec![],
            next_cursor: None,
        })
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, McpError> {
        Err(McpError::resource_not_found(
            format!("Resource URI not found: {}", request.uri),
            Some(json!({ "uri": request.uri })),
        ))
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
