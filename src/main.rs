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
    server::{RustDocsServer, CrateData}, // Import the updated RustDocsServer and CrateData
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
    collections::{hash_map::DefaultHasher, HashMap},
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
    /// The package ID specifications with optional features (e.g., "serde@^1.0:feature1:feature2", "tokio", "reqwest@0.12:json").
    #[arg(required = true)] // Positional arguments, at least one required
    package_specs: Vec<String>,
}

#[derive(Debug, Clone)]
struct CrateSpec {
    name: String,
    version_req: String,
    features: Option<Vec<String>>,
    original_spec: String,
}

// Helper function to parse crate specification with features
fn parse_crate_spec(spec: &str) -> Result<CrateSpec, String> {
    // Split by ':' to separate crate spec from features
    let parts: Vec<&str> = spec.split(':').collect();
    
    if parts.is_empty() {
        return Err("Empty crate specification".to_string());
    }
    
    let crate_part = parts[0];
    let features = if parts.len() > 1 {
        Some(parts[1..].iter().map(|s| s.to_string()).collect())
    } else {
        None
    };
    
    // Parse the crate part using PackageIdSpec
    let package_spec = PackageIdSpec::parse(crate_part).map_err(|e| {
        format!("Failed to parse package ID spec '{}': {}", crate_part, e)
    })?;
    
    let name = package_spec.name().to_string();
    let version_req = package_spec
        .version()
        .map(|v| v.to_string())
        .unwrap_or_else(|| "*".to_string());
    
    Ok(CrateSpec {
        name,
        version_req,
        features,
        original_spec: spec.to_string(),
    })
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
    
    // Parse all package specs with features
    let mut parsed_crates = Vec::new();
    for spec_str in &cli.package_specs {
        let crate_spec = parse_crate_spec(spec_str.trim()).map_err(|e| {
            ServerError::Config(e)
        })?;
        parsed_crates.push(crate_spec);
    }

    eprintln!("Target Crates:");
    for crate_spec in &parsed_crates {
        let features_str = crate_spec.features.as_ref()
            .map(|f| format!(" [features: {}]", f.join(", ")))
            .unwrap_or_default();
        eprintln!("  - {}@{}{}", crate_spec.name, crate_spec.version_req, features_str);
    }

    // --- Initialize OpenAI Client (needed for question embedding even if cache hit) ---
    let openai_client = if let Ok(api_base) = env::var("OPENAI_API_BASE") {
        let config = OpenAIConfig::new().with_api_base(api_base);
        OpenAIClient::with_config(config)
    } else {
        OpenAIClient::new()
    };
    OPENAI_CLIENT
        .set(openai_client.clone()) // Clone the client for the OnceCell
        .expect("Failed to set OpenAI client");

    // Check for API key
    let _openai_api_key = env::var("OPENAI_API_KEY")
        .map_err(|_| ServerError::MissingEnvVar("OPENAI_API_KEY".to_string()))?;

    // --- Process Each Crate ---
    let mut crates_data = HashMap::new();
    let mut all_loaded_from_cache = Vec::new();
    let mut total_generated_tokens = 0;
    let mut total_generation_cost = 0.0;

    for crate_spec in &parsed_crates {
        let crate_name = &crate_spec.name;
        let crate_version_req = &crate_spec.version_req;
        let features = &crate_spec.features;
        
        eprintln!("Processing crate: {} (Version Req: {}, Features: {:?})", crate_name, crate_version_req, features);

        // --- Determine Paths (incorporating per-crate features) ---
        // Generate a stable hash for this crate's features
        let features_hash = hash_features(features);
        
        // Sanitize the version requirement string
        let sanitized_version_req = crate_version_req
            .replace(|c: char| !c.is_alphanumeric() && c != '.' && c != '-', "_");

        // Construct the relative path component including features hash
        let embeddings_relative_path = PathBuf::from(crate_name)
            .join(&sanitized_version_req)
            .join(&features_hash) // Add features hash as a directory level
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
            // Ensure the base app cache directory exists
            fs::create_dir_all(&app_cache_dir).map_err(ServerError::Io)?;
            app_cache_dir.join(embeddings_relative_path)
        };

        eprintln!("Cache file path for {}: {:?}", crate_name, embeddings_file_path);

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

        let final_embeddings = match loaded_embeddings {
            Some(embeddings) => {
                eprintln!("Using embeddings and documents loaded from cache for {}.", crate_name);
                all_loaded_from_cache.push(crate_name.clone());
                embeddings
            }
            None => {
                eprintln!("Proceeding with documentation loading and embedding generation for {}.", crate_name);

                eprintln!(
                    "Loading documents for crate: {} (Version Req: {}, Features: {:?})",
                    crate_name, crate_version_req, features
                );
                // Pass features to load_documents
                let loaded_documents =
                    doc_loader::load_documents(crate_name, crate_version_req, features.as_ref())?;
                eprintln!("Loaded {} documents for {}.", loaded_documents.len(), crate_name);
                documents_for_server = loaded_documents.clone();

                eprintln!("Generating embeddings for {}...", crate_name);
                let embedding_model: String = env::var("EMBEDDING_MODEL")
                    .unwrap_or_else(|_| "text-embedding-3-small".to_string());
                let (generated_embeddings, total_tokens) =
                    generate_embeddings(&openai_client, &loaded_documents, &embedding_model).await?;

                let cost_per_million = 0.02;
                let estimated_cost = (total_tokens as f64 / 1_000_000.0) * cost_per_million;
                eprintln!(
                    "Embedding generation cost for {} ({} tokens): ${:.6}",
                    crate_name, total_tokens, estimated_cost
                );
                generated_tokens = Some(total_tokens);
                generation_cost = Some(estimated_cost);
                total_generated_tokens += total_tokens;
                total_generation_cost += estimated_cost;

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

        // Create metadata string for this crate
        let metadata = if loaded_from_cache {
            format!("Version: {}, Features: {:?}, Loaded from cache", crate_version_req, features)
        } else {
            let tokens = generated_tokens.unwrap_or(0);
            let cost = generation_cost.unwrap_or(0.0);
            format!("Version: {}, Features: {:?}, Generated {} embeddings for {} tokens (Cost: ${:.6})", 
                    crate_version_req, features, final_embeddings.len(), tokens, cost)
        };

        // Store crate data
        crates_data.insert(crate_name.clone(), CrateData {
            documents: documents_for_server,
            embeddings: final_embeddings,
            metadata,
        });

        eprintln!("Completed processing crate: {}", crate_name);
    }

    // --- Create startup message for all crates ---
    let startup_message = if all_loaded_from_cache.len() == parsed_crates.len() {
        format!(
            "Server initialized with {} crates: {}. All loaded from cache.",
            parsed_crates.len(),
            parsed_crates.iter().map(|spec| spec.name.as_str()).collect::<Vec<_>>().join(", ")
        )
    } else {
        format!(
            "Server initialized with {} crates: {}. Generated {} total tokens (Est. Cost: ${:.6}).",
            parsed_crates.len(),
            parsed_crates.iter().map(|spec| spec.name.as_str()).collect::<Vec<_>>().join(", "),
            total_generated_tokens,
            total_generation_cost
        )
    };

    // Create the service instance using the updated ::new()
    let service = RustDocsServer::new(
        crates_data,
        startup_message,
    )?;

    // --- Use standard stdio transport and ServiceExt ---
    eprintln!("Rust Docs MCP server starting via stdio...");

    // Serve the server using the ServiceExt trait and standard stdio transport
    let server_handle = service.serve(stdio()).await.map_err(|e| {
        eprintln!("Failed to start server: {:?}", e);
        ServerError::McpRuntime(e.to_string()) // Use the new McpRuntime variant
    })?;

    eprintln!("Multi-crate Docs MCP server running...");

    // Wait for the server to complete (e.g., stdin closed)
    server_handle.waiting().await.map_err(|e| {
        eprintln!("Server encountered an error while running: {:?}", e);
        ServerError::McpRuntime(e.to_string()) // Use the new McpRuntime variant
    })?;

    eprintln!("Rust Docs MCP server stopped.");
    Ok(())
}
