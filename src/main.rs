// Declare modules (keep doc_loader, embeddings, error)
mod doc_loader;
mod embeddings;
mod error;
mod server; // Keep server module as RustDocsServer is defined there

// Use necessary items from modules and crates
use crate::{
    // doc_loader::DocumentChunk, // Unused
    embeddings::{CachedChunkEmbedding, OPENAI_CLIENT}, // Removed unused generate_embeddings
    error::ServerError,
    server::RustDocsServer,
};
use async_openai::Client as OpenAIClient;
use bincode::config;
use cargo::core::PackageIdSpec;
use clap::Parser; // Import clap Parser
// use ndarray::Array1; // Remove unused Array1
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
    #[arg()] // Positional argument
    package_spec: String,

    /// Optional features to enable for the crate when generating documentation.
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
    let specid_str = cli.package_spec.trim().to_string(); // Trim whitespace
    let features = cli.features.map(|f| {
        f.into_iter().map(|s| s.trim().to_string()).collect() // Trim each feature
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
        "Target Spec: {}, Parsed Name: {}, Version Req: {}, Features: {:?}",
        specid_str, crate_name, crate_version_req, features
    );

    // --- Determine Paths (incorporating features) ---

    // Sanitize the version requirement string
    let sanitized_version_req = crate_version_req
        .replace(|c: char| !c.is_alphanumeric() && c != '.' && c != '-', "_");

    // Generate a stable hash for the features to use in the path
    let features_hash = hash_features(&features);

    // Construct the relative path component including features hash
    let embeddings_relative_path = PathBuf::from(&crate_name)
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

    eprintln!("Cache file path: {:?}", embeddings_file_path);

    // --- Try Loading Embeddings and Documents from Cache ---
    let mut loaded_from_cache = false;
    // We'll load the combined chunk data directly
    let mut loaded_cached_chunks: Option<Vec<CachedChunkEmbedding>> = None;

    if embeddings_file_path.exists() {
        eprintln!(
            "Attempting to load cached data from: {:?}",
            embeddings_file_path
        );
        match File::open(&embeddings_file_path) {
            Ok(file) => {
                let reader = BufReader::new(file);
                // Decode directly into the target type Vec<CachedChunkEmbedding>
                match bincode::decode_from_reader::<Vec<CachedChunkEmbedding>, _, _>(
                    reader,
                    config::standard(),
                ) {
                    Ok(cached_chunks) => {
                        eprintln!(
                            "Successfully loaded {} cached chunks from cache.",
                            cached_chunks.len()
                        );
                        // Store the loaded chunks directly
                        loaded_cached_chunks = Some(cached_chunks);
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
    // We'll store the final list of chunks (either loaded or generated) here
    let final_cached_chunks: Vec<CachedChunkEmbedding>;
    // Removed unused variables related to cost calculation
    // let mut generated_tokens: Option<usize> = None;
    // let mut generation_cost: Option<f64> = None;

    // --- Initialize OpenAI Client (needed for question embedding even if cache hit) ---
    let openai_client = OpenAIClient::new();
    OPENAI_CLIENT
        .set(openai_client.clone()) // Clone the client for the OnceCell
        .expect("Failed to set OpenAI client");

    // Decide whether to use cached chunks or generate new ones
    if let Some(cached_chunks) = loaded_cached_chunks {
         eprintln!("Using {} chunks loaded from cache.", cached_chunks.len());
         final_cached_chunks = cached_chunks;
        } // End: if let Some(cached_chunks) block
    // Corrected 'else' block for cache miss, associated with the 'if let' on line 184
    else {
        // --- Generate Docs & Embeddings (Cache Miss) ---
        eprintln!("Cache miss. Proceeding with documentation loading and embedding generation.");

        // Ensure OpenAI key is available before proceeding
        let _openai_api_key = env::var("OPENAI_API_KEY")
            .map_err(|_| ServerError::MissingEnvVar("OPENAI_API_KEY".to_string()))?;

        eprintln!(
            "Loading and chunking documents for crate: {} (Version Req: {}, Features: {:?})",
            crate_name, crate_version_req, features
        );
        // Load and chunk documents - returns Result<Vec<DocumentChunk>, DocLoaderError>
        let loaded_chunks =
            doc_loader::load_documents(&crate_name, &crate_version_req, features.as_ref())?;
        eprintln!("Created {} document chunks.", loaded_chunks.len());

        // Check if chunks were actually created before trying to embed
        if loaded_chunks.is_empty() {
             eprintln!("No document chunks were created. Proceeding without embeddings.");
             // Set final_cached_chunks to empty if no source chunks were loaded
             final_cached_chunks = Vec::new();
        } else {
            eprintln!("Generating embeddings for {} chunks...", loaded_chunks.len());
            // Generate embeddings for the loaded chunks
            // generate_embeddings now returns Result<(Vec<CachedChunkEmbedding>, usize), ServerError>
            // generate_embeddings now only returns the Vec of embeddings
            let generated_cached_chunks =
                embeddings::generate_embeddings(&loaded_chunks).await?;
            // let total_tokens = 0; // Removed - cost calculation skipped for now

            eprintln!("Successfully generated {} embeddings.", generated_cached_chunks.len());
            // generated_tokens = Some(0); // Removed - No longer tracking tokens
            // Assign the generated chunks to final_cached_chunks
            final_cached_chunks = generated_cached_chunks;

            // Calculate cost (example for text-embedding-3-small)
            // let cost_per_1k_tokens = 0.00002; // Removed
            // Cost calculation removed
            // generation_cost = Some(cost);
            // eprintln!( // Removed cost printing
            //     "Estimated OpenAI Embedding Cost: ~${:.6}",
            //     cost
            // );

            // --- Save Generated Chunks to Cache ---
            // Only attempt to save if embeddings were actually generated
            if !final_cached_chunks.is_empty() {
                eprintln!("Saving {} generated chunks to cache: {:?}", final_cached_chunks.len(), embeddings_file_path);
                // Ensure parent directory exists
                if let Some(parent_dir) = embeddings_file_path.parent() {
                    fs::create_dir_all(parent_dir).map_err(ServerError::Io)?;
                }
                // Encode data to Vec<u8> first
                match bincode::encode_to_vec(&final_cached_chunks, config::standard()) {
                    Ok(encoded_bytes) => {
                        // Try to write the encoded bytes to the file
                        match fs::write(&embeddings_file_path, encoded_bytes) {
                            Ok(_) => eprintln!("Successfully saved data to cache."),
                            Err(e) => eprintln!("Warning: Failed to write cache file: {}", e), // Log file write error
                        }
                    }
                    Err(e) => eprintln!("Warning: Failed to encode cache data: {}", e), // Log encoding error
                }
            } else {
                 eprintln!("Skipping cache save as no embeddings were generated.");
            }
        }
    } // End cache miss 'else' block


    // --- Prepare Server Startup Message ---
    // --- Prepare Server Startup Message ---
    let startup_message = format!( // Removed mut
        "RustDocs MCP Server Initialized.\nCrate: {}\nVersion Req: {}\nFeatures: {:?}\nCache Used: {}\n{} Chunks Ready.",
        crate_name,
        crate_version_req,
        features.as_deref().unwrap_or(&[]), // Display features nicely
        if loaded_from_cache { "Yes" } else { "No" },
        final_cached_chunks.len() // Use length of final chunks
    );
    // Removed cost message appending from startup message
    // if let Some(cost) = generation_cost {
    //     let cost_msg = format!("\nGeneration Cost: ~${:.6} ({} tokens)", cost, generated_tokens.unwrap_or(0));
    //     startup_message.push_str(&cost_msg);
    // }

     // --- Start Server ---
    eprintln!("{}", startup_message); // Log startup message to stderr
    // Create the service instance using the final cached chunks
    let service = RustDocsServer::new(
        crate_name.clone(),
        final_cached_chunks, // Pass the final Vec<CachedChunkEmbedding>
        startup_message,     // Pass the generated startup message
    )?; // Propagate error from new() if needed

    // --- Use standard stdio transport and ServiceExt ---
    eprintln!("Rust Docs MCP server starting via stdio...");

    // Serve the server using the ServiceExt trait and standard stdio transport
    let server_handle = service.serve(stdio()).await.map_err(|e| {
        eprintln!("Failed to start server: {:?}", e);
        ServerError::McpRuntime(e.to_string()) // Use the new McpRuntime variant
    })?;

    eprintln!("{} Docs MCP server running...", &crate_name);

    // Wait for the server to complete (e.g., stdin closed)
    server_handle.waiting().await.map_err(|e| {
        eprintln!("Server encountered an error while running: {:?}", e);
        ServerError::McpRuntime(e.to_string()) // Use the new McpRuntime variant
    })?;

    eprintln!("Rust Docs MCP server stopped.");
    Ok(())
}
