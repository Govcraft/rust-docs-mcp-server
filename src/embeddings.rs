// Keep necessary imports, remove duplicates later
use crate::doc_loader::DocumentChunk; // Use DocumentChunk
use crate::error::ServerError;
use async_openai::{
    config::OpenAIConfig,
    // error::ApiError as OpenAIAPIErr, // Unused
    // types::{CreateEmbeddingRequestArgs}, // Unused
    Client as OpenAIClient,
};
use bincode::{Decode, Encode}; // Used in derive
use futures::stream::{self, StreamExt, TryStreamExt};
use ndarray::ArrayView1; // Keep ArrayView1
use serde::{Deserialize, Serialize}; // Keep serde imports for the struct derive
use std::sync::{Arc, OnceLock}; // Combine Arc and OnceLock import
use tiktoken_rs::cl100k_base;

// Define the OpenAI client static variable ONCE
pub static OPENAI_CLIENT: OnceLock<OpenAIClient<OpenAIConfig>> = OnceLock::new();


// Remove duplicate imports (lines 16-27 were duplicates)

/// Represents a pre-computed embedding for a single documentation chunk, ready for caching.
// Add Encode and Decode back, needed for caching
#[derive(Serialize, Deserialize, Debug, Clone, Encode, Decode)]
pub struct CachedChunkEmbedding {
    /// Relative path of the original source file (e.g., "struct.MyStruct.html").
    pub source_path: String,
    /// The 0-based index of this chunk within its original source file.
    pub chunk_index: usize,
    /// The actual text content of this chunk. Stored for context retrieval.
    pub content: String,
    /// The embedding vector for this chunk's content. Store as Vec<f32> for easier serialization.
    pub vector: Vec<f32>,
}

// --- Constants ---
const CONCURRENCY_LIMIT: usize = 10; // Max concurrent OpenAI requests
const TOKEN_LIMIT: usize = 8191; // Max tokens for text-embedding-3-small
const EMBEDDING_MODEL: &str = "text-embedding-3-small"; // Define model constant

// Remove duplicate static definition (lines 48-50)

// --- Embedding Generation ---

/// Generates embeddings for a list of document chunks using the OpenAI API.
///
/// Skips chunks whose content exceeds the `TOKEN_LIMIT`.
/// Returns a vector of successfully generated `CachedChunkEmbedding`s.
// Keep only one definition of generate_embeddings
pub async fn generate_embeddings(
    documents: &[DocumentChunk],
) -> Result<Vec<CachedChunkEmbedding>, ServerError> { // Removed usize from return tuple
    let client = OPENAI_CLIENT
        .get()
        .expect("OpenAI client should be initialized");

    // Wrap BPE tokenizer in Arc for sharing across async tasks
    let bpe = Arc::new(cl100k_base().map_err(|e| ServerError::Tiktoken(e.to_string()))?);

    // Use try_filter_map for cleaner error handling within the stream
    let results = stream::iter(documents)
        .map(|doc| {
            // Clone client reference for the async block
            let client_ref = client.clone();
            // Clone the doc and bpe needed inside the async block
            let doc_clone = doc.clone();
            let bpe_arc_clone = Arc::clone(&bpe); // Clone the Arc pointer
            async move {
                // Calculate token count first, propagating potential error
                // Calculate token count first, handling potential error explicitly
                // Calculate token count first, handling potential error explicitly
                // Calculate token count first, handling potential error explicitly using ?
                // Calculate token count first, handling potential error explicitly
                // Calculate token count first, handling potential error explicitly with match
                // Calculate token count first, handling potential error explicitly with match
                // Calculate token count first, handling potential error explicitly using map_err and ?
                // Calculate token count first, handling potential error explicitly with match
                // Declare tokens variable first
                // Handle tokenization result using if let/else
                // Calculate token count first, handling potential error explicitly using map_err and ?
                // Assume encode_with_special_tokens returns Vec<u32> directly based on compiler errors.
                // NOTE: This means potential panics from tiktoken-rs are not caught here.
                let tokens = bpe_arc_clone.encode_with_special_tokens(&doc_clone.content); // Use the Arc clone
                // Now 'tokens' holds Vec<u32>

                // Check token limit
                if tokens.len() > TOKEN_LIMIT {
                     eprintln!(
                         "Skipping chunk {} from '{}' due to token limit ({} > {})",
                         doc_clone.chunk_index, doc_clone.source_path, tokens.len(), TOKEN_LIMIT
                     );
                     // Explicitly state the full Result type including the Error variant
                     return Ok::<Option<(DocumentChunk, Vec<f32>, usize)>, ServerError>(None);
                 }

                // Proceed with embedding request
                let request = async_openai::types::CreateEmbeddingRequestArgs::default()
                    .model(EMBEDDING_MODEL)
                    .input(vec![doc_clone.content.clone()])
                    .build()
                    .map_err(ServerError::OpenAI)?;

                let response = client_ref.embeddings().create(request).await.map_err(ServerError::OpenAI)?;

                let embedding_vector = response.data.into_iter().next()
                    .ok_or_else(|| ServerError::OpenAI(async_openai::error::OpenAIError::ApiError(
                        async_openai::error::ApiError {
                            message: format!("No embedding returned for chunk {} from '{}'", doc_clone.chunk_index, doc_clone.source_path),
                            r#type: Some("embedding_error".to_string()),
                            param: None, code: None,
                        }
                    )))?.embedding;

                // Return successful result
                Ok(Some((doc_clone, embedding_vector, tokens.len())))
                // Removed duplicated code block
            }
        })
        .buffer_unordered(CONCURRENCY_LIMIT)
        .try_collect::<Vec<Option<(DocumentChunk, Vec<f32>, usize)>>>() // Add type annotation back
        .await?; // Add back the ? operator here

    // Filter out None values (skipped chunks) and calculate total tokens
    // let mut total_tokens = 0; // Removed token counting
    // Explicitly type results after awaiting
    let results: Vec<Option<(DocumentChunk, Vec<f32>, usize)>> = results;

    // Use iterator chain again for filtering and mapping
    let cached_embeddings: Vec<CachedChunkEmbedding> = results
        .into_iter()
        .filter_map(|opt_result| opt_result) // Filter out None values
        .map(|(doc, vector, _count)| { // Map the Some(tuple) to CachedChunkEmbedding
            // total_tokens += count; // Removed token counting
            CachedChunkEmbedding {
                source_path: doc.source_path,
                chunk_index: doc.chunk_index,
                content: doc.content,
                vector, // Store the Vec<f32> directly
            }
        })
        .collect();

    Ok(cached_embeddings) // Return only the embeddings vector
}

// --- Cosine Similarity ---

// Keep only one definition of cosine_similarity
/// Calculates the cosine similarity between two vectors.
pub fn cosine_similarity(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
    let dot_product = a.dot(&b);
    let norm_a = a.dot(&a).sqrt();
    let norm_b = b.dot(&b).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0 // Avoid division by zero
    } else {
        dot_product / (norm_a * norm_b)
    }
}
// Remove duplicate cosine_similarity definition (lines 212-221)
// Remove duplicate generate_embeddings definition (lines 225-328)


// Remove the serde_array module as it's no longer needed


// Removed duplicate definitions