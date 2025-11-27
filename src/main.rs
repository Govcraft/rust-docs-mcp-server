// Declare modules (keep doc_loader, embeddings, error)
mod doc_loader;
mod embeddings;
mod error;
mod server; // Keep server module as RustDocsServer is defined there

// Use necessary items from modules and crates
use crate::{
    doc_loader::Document,
    embeddings::{generate_embeddings, CachedDocumentEmbedding, OPENAI_CLIENT},
    error::ServerError,
};
use async_openai::{Client as OpenAIClient, config::OpenAIConfig};
use bincode::config;
use cargo::core::PackageIdSpec;
use clap::Parser; // Import clap Parser
use ndarray::Array1;
// Import rmcp items needed for the new approach
use rmcp::{
    transport::io::stdio, // Use the standard stdio transport
    ServiceExt,           // Import the ServiceExt trait for .serve() and .waiting()
};
use std::{
    collections::hash_map::DefaultHasher,
    env,
    fs::{self, File},
    hash::{Hash, Hasher}, // Import hashing utilities
    io::BufReader,
    path::PathBuf,
};
#[cfg(not(target_os = "windows"))]
use xdg::BaseDirectories;

// --- CLI Argument Parsing ---

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// The package ID specification (e.g., "serde@^1.0", "tokio").
    /// If not provided, server runs in "any crate mode" allowing queries for any crate.
    #[arg()] // Positional argument
    package_spec: Option<String>,

    /// Optional features to enable for the crate when generating documentation.
    /// Only used in single-crate mode.
    #[arg(short = 'F', long, value_delimiter = ',', num_args = 0..)] // Allow multiple comma-separated values
    features: Option<Vec<String>>,
}

// Helper function to create a stable hash from features
fn hash_features(features: &Option<Vec<String>>) -> String {
    features
        .as_ref()
        .map(|f| {
            let mut sorted_features = f.clone();
            sorted_features.sort_unstable(); // Sort for consistent hashing
            let mut hasher = DefaultHasher::new();
            sorted_features.hash(&mut hasher);
            format!("{:x}", hasher.finish()) // Return hex representation of hash
        })
        .unwrap_or_else(|| "no_features".to_string()) // Use a specific string if no features
}

#[tokio::main]
async fn main() -> Result<(), ServerError> {
    // Load .env file if present
    dotenvy::dotenv().ok();

    // --- Parse CLI Arguments ---
    let cli = Cli::parse();

    // Initialize OpenAI Client early (needed for both modes)
    let openai_client = if let Ok(api_base) = env::var("OPENAI_API_BASE") {
        let config = OpenAIConfig::new().with_api_base(api_base);
        OpenAIClient::with_config(config)
    } else {
        OpenAIClient::new()
    };
    OPENAI_CLIENT
        .set(openai_client.clone())
        .expect("Failed to set OpenAI client");

    // Determine which mode to run in
    match cli.package_spec {
        Some(package_spec) => {
            // Single-crate mode
            run_single_crate_mode(package_spec, cli.features).await
        }
        None => {
            // Any-crate mode
            run_any_crate_mode().await
        }
    }
}

/// Run the server in single-crate mode with pre-loaded documentation
async fn run_single_crate_mode(
    package_spec: String,
    features: Option<Vec<String>>,
) -> Result<(), ServerError> {
    let specid_str = package_spec.trim().to_string();
    let features = features.map(|f| {
        f.into_iter().map(|s| s.trim().to_string()).collect()
    });

    // Parse the specid string
    let spec = PackageIdSpec::parse(&specid_str).map_err(|e| {
        ServerError::Config(format!(
            "Failed to parse package ID spec '{}': {}",
            specid_str, e
        ))
    })?;

    let crate_name = spec.name().to_string();
    let crate_version_req = spec
        .version()
        .map(|v| v.to_string())
        .unwrap_or_else(|| "*".to_string());

    eprintln!(
        "Single-crate mode - Target Spec: {}, Parsed Name: {}, Version Req: {}, Features: {:?}",
        specid_str, crate_name, crate_version_req, features
    );

    // --- Determine Paths (incorporating features) ---
    let sanitized_version_req = crate_version_req
        .replace(|c: char| !c.is_alphanumeric() && c != '.' && c != '-', "_");

    let features_hash = hash_features(&features);

    let embeddings_relative_path = PathBuf::from(&crate_name)
        .join(&sanitized_version_req)
        .join(&features_hash)
        .join("embeddings.bin");

    #[cfg(not(target_os = "windows"))]
    let embeddings_file_path = {
        let xdg_dirs = BaseDirectories::with_prefix("rustdocs-mcp-server")
            .map_err(|e| ServerError::Xdg(format!("Failed to get XDG directories: {}", e)))?;
        xdg_dirs
            .place_data_file(embeddings_relative_path)
            .map_err(ServerError::Io)?
    };

    #[cfg(target_os = "windows")]
    let embeddings_file_path = {
        let cache_dir = dirs::cache_dir().ok_or_else(|| {
            ServerError::Config("Could not determine cache directory on Windows".to_string())
        })?;
        let app_cache_dir = cache_dir.join("rustdocs-mcp-server");
        fs::create_dir_all(&app_cache_dir).map_err(ServerError::Io)?;
        app_cache_dir.join(embeddings_relative_path)
    };

    eprintln!("Cache file path: {:?}", embeddings_file_path);

    // --- Try Loading Embeddings and Documents from Cache ---
    let mut loaded_from_cache = false;
    let mut loaded_embeddings: Option<Vec<(String, Array1<f32>)>> = None;
    let mut loaded_documents_from_cache: Option<Vec<Document>> = None;

    if embeddings_file_path.exists() {
        eprintln!(
            "Attempting to load cached data from: {:?}",
            embeddings_file_path
        );
        match File::open(&embeddings_file_path) {
            Ok(file) => {
                let reader = BufReader::new(file);
                match bincode::decode_from_reader::<Vec<CachedDocumentEmbedding>, _, _>(
                    reader,
                    config::standard(),
                ) {
                    Ok(cached_data) => {
                        eprintln!(
                            "Successfully loaded {} items from cache. Separating data...",
                            cached_data.len()
                        );
                        let mut embeddings = Vec::with_capacity(cached_data.len());
                        let mut documents = Vec::with_capacity(cached_data.len());
                        for item in cached_data {
                            embeddings.push((item.path.clone(), Array1::from(item.vector)));
                            documents.push(Document {
                                path: item.path,
                                content: item.content,
                            });
                        }
                        loaded_embeddings = Some(embeddings);
                        loaded_documents_from_cache = Some(documents);
                        loaded_from_cache = true;
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
    } else {
        eprintln!("Cache file not found. Will generate.");
    }

    // --- Generate or Use Loaded Embeddings ---
    let mut generated_tokens: Option<usize> = None;
    let mut generation_cost: Option<f64> = None;
    let mut documents_for_server: Vec<Document> = loaded_documents_from_cache.unwrap_or_default();

    let openai_client = OPENAI_CLIENT.get().unwrap();

    let final_embeddings = match loaded_embeddings {
        Some(embeddings) => {
            eprintln!("Using embeddings and documents loaded from cache.");
            embeddings
        }
        None => {
            eprintln!("Proceeding with documentation loading and embedding generation.");

            let _openai_api_key = env::var("OPENAI_API_KEY")
                .map_err(|_| ServerError::MissingEnvVar("OPENAI_API_KEY".to_string()))?;

            eprintln!(
                "Loading documents for crate: {} (Version Req: {}, Features: {:?})",
                crate_name, crate_version_req, features
            );
            let loaded_documents =
                doc_loader::load_documents(&crate_name, &crate_version_req, features.as_ref())?;
            eprintln!("Loaded {} documents.", loaded_documents.len());
            documents_for_server = loaded_documents.clone();

            eprintln!("Generating embeddings...");
            let embedding_model: String = env::var("EMBEDDING_MODEL")
                .unwrap_or_else(|_| "text-embedding-3-small".to_string());
            let (generated_embeddings, total_tokens) =
                generate_embeddings(&openai_client, &loaded_documents, &embedding_model).await?;

            let cost_per_million = 0.02;
            let estimated_cost = (total_tokens as f64 / 1_000_000.0) * cost_per_million;
            eprintln!(
                "Embedding generation cost for {} tokens: ${:.6}",
                total_tokens, estimated_cost
            );
            generated_tokens = Some(total_tokens);
            generation_cost = Some(estimated_cost);

            eprintln!(
                "Saving generated documents and embeddings to: {:?}",
                embeddings_file_path
            );

            let mut combined_cache_data: Vec<CachedDocumentEmbedding> = Vec::new();
            let embedding_map: std::collections::HashMap<String, Array1<f32>> =
                generated_embeddings.clone().into_iter().collect();

            for doc in &loaded_documents {
                if let Some(embedding_array) = embedding_map.get(&doc.path) {
                    combined_cache_data.push(CachedDocumentEmbedding {
                        path: doc.path.clone(),
                        content: doc.content.clone(),
                        vector: embedding_array.to_vec(),
                    });
                } else {
                    eprintln!(
                        "Warning: Embedding not found for document path: {}. Skipping from cache.",
                        doc.path
                    );
                }
            }

            match bincode::encode_to_vec(&combined_cache_data, config::standard()) {
                Ok(encoded_bytes) => {
                    if let Some(parent_dir) = embeddings_file_path.parent() {
                        if !parent_dir.exists() {
                            if let Err(e) = fs::create_dir_all(parent_dir) {
                                eprintln!(
                                    "Warning: Failed to create cache directory {}: {}",
                                    parent_dir.display(),
                                    e
                                );
                            }
                        }
                    }
                    if let Err(e) = fs::write(&embeddings_file_path, encoded_bytes) {
                        eprintln!("Warning: Failed to write cache file: {}", e);
                    } else {
                        eprintln!(
                            "Cache saved successfully ({} items).",
                            combined_cache_data.len()
                        );
                    }
                }
                Err(e) => {
                    eprintln!("Warning: Failed to encode data for cache: {}", e);
                }
            }
            generated_embeddings
        }
    };

    // --- Initialize and Start Server ---
    eprintln!(
        "Initializing server for crate: {} (Version Req: {}, Features: {:?})",
        crate_name, crate_version_req, features
    );

    let features_str = features
        .as_ref()
        .map(|f| format!(" Features: {:?}", f))
        .unwrap_or_default();

    let startup_message = if loaded_from_cache {
        format!(
            "Server for crate '{}' (Version Req: '{}'{}) initialized. Loaded {} embeddings from cache.",
            crate_name, crate_version_req, features_str, final_embeddings.len()
        )
    } else {
        let tokens = generated_tokens.unwrap_or(0);
        let cost = generation_cost.unwrap_or(0.0);
        format!(
            "Server for crate '{}' (Version Req: '{}'{}) initialized. Generated {} embeddings for {} tokens (Est. Cost: ${:.6}).",
            crate_name,
            crate_version_req,
            features_str,
            final_embeddings.len(),
            tokens,
            cost
        )
    };

    // Create the service instance for single-crate mode
    let service = server::RustDocsSingleCrateServer::new(
        crate_name.clone(),
        documents_for_server,
        final_embeddings,
        startup_message,
    )?;

    // --- Use standard stdio transport and ServiceExt ---
    eprintln!("Rust Docs MCP server starting via stdio (single-crate mode)...");

    let server_handle = service.serve(stdio()).await.map_err(|e| {
        eprintln!("Failed to start server: {:?}", e);
        ServerError::McpRuntime(e.to_string())
    })?;

    eprintln!("{} Docs MCP server running...", &crate_name);

    server_handle.waiting().await.map_err(|e| {
        eprintln!("Server encountered an error while running: {:?}", e);
        ServerError::McpRuntime(e.to_string())
    })?;

    eprintln!("Rust Docs MCP server stopped.");
    Ok(())
}

/// Run the server in any-crate mode where documentation is loaded on-demand
async fn run_any_crate_mode() -> Result<(), ServerError> {
    eprintln!("Any-crate mode - Server will load documentation on demand for any requested crate");

    // Verify OpenAI API key is available
    let _openai_api_key = env::var("OPENAI_API_KEY")
        .map_err(|_| ServerError::MissingEnvVar("OPENAI_API_KEY".to_string()))?;

    // Create the service instance for any-crate mode
    let service = server::RustDocsAnyCrateServer::new()?;

    // --- Use standard stdio transport and ServiceExt ---
    eprintln!("Rust Docs MCP server starting via stdio (any-crate mode)...");

    let server_handle = service.serve(stdio()).await.map_err(|e| {
        eprintln!("Failed to start server: {:?}", e);
        ServerError::McpRuntime(e.to_string())
    })?;

    eprintln!("Rust Docs MCP server running (any-crate mode)...");

    server_handle.waiting().await.map_err(|e| {
        eprintln!("Server encountered an error while running: {:?}", e);
        ServerError::McpRuntime(e.to_string())
    })?;

    eprintln!("Rust Docs MCP server stopped.");
    Ok(())
}
