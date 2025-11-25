use crate::{doc_loader::Document, error::ServerError};
use async_openai::{
    config::OpenAIConfig, error::ApiError as OpenAIAPIErr, types::CreateEmbeddingRequestArgs,
    Client as OpenAIClient,
};
use ndarray::{Array1, ArrayView1};
use std::sync::OnceLock;
use std::sync::Arc;
use tiktoken_rs::cl100k_base;
use futures::stream::{self, StreamExt};

// Static OnceLock for the OpenAI client
pub static OPENAI_CLIENT: OnceLock<OpenAIClient<OpenAIConfig>> = OnceLock::new();


use bincode::{Encode, Decode};
use serde::{Serialize, Deserialize};

// Define a struct containing path, content, and embedding for caching
#[derive(Serialize, Deserialize, Debug, Encode, Decode)]
pub struct CachedDocumentEmbedding {
    pub path: String,
    pub content: String, // Add the extracted document content
    pub vector: Vec<f32>,
}


/// Calculates the cosine similarity between two vectors.
pub fn cosine_similarity(v1: ArrayView1<f32>, v2: ArrayView1<f32>) -> f32 {
    let dot_product = v1.dot(&v2);
    let norm_v1 = v1.dot(&v1).sqrt();
    let norm_v2 = v2.dot(&v2).sqrt();

    if norm_v1 == 0.0 || norm_v2 == 0.0 {
        0.0
    } else {
        dot_product / (norm_v1 * norm_v2)
    }
}

/// Generates embeddings for a list of documents using the OpenAI API.
pub async fn generate_embeddings(
    client: &OpenAIClient<OpenAIConfig>,
    documents: &[Document],
    model: &str,
) -> Result<(Vec<(String, Array1<f32>)>, usize), ServerError> { // Return tuple: (embeddings, total_tokens)
    // eprintln!("Generating embeddings for {} documents...", documents.len());

    // Get the tokenizer for the model and wrap in Arc
    let bpe = Arc::new(cl100k_base().map_err(|e| ServerError::Tiktoken(e.to_string()))?);

    const CONCURRENCY_LIMIT: usize = 8; // Number of concurrent requests

    // Our default model only supports roughly 8k tokens
    let token_limit: usize = std::env::var("EMBEDDING_TOKEN_LIMIT")
        .ok()
        .and_then(|lim| lim.trim().parse().ok())
        .unwrap_or(8000);

    let results = stream::iter(documents.iter().enumerate())
        .map(|(index, doc)| {
            // Clone client, model, doc, and Arc<BPE> for the async block
            let client = client.clone();
            let model = model.to_string();
            let doc = doc.clone();
            let bpe = Arc::clone(&bpe); // Clone the Arc pointer

            async move {
                // Calculate token count for this document
                let token_count = bpe.encode_with_special_tokens(&doc.content).len();

                if token_count > token_limit {
                    // eprintln!(
                    //     "    Skipping document {}: Actual tokens ({}) exceed limit ({}). Path: {}",
                    //     index + 1,
                    //     token_count,
                    //     token_limit,
                    //     doc.path
                    // );
                    // Return Ok(None) to indicate skipping, with 0 tokens processed for this doc
                    return Ok::<Option<(String, Array1<f32>, usize)>, ServerError>(None); // Include token count type
                }

                // Prepare input for this single document
                let inputs: Vec<String> = vec![doc.content.clone()];

                let request = CreateEmbeddingRequestArgs::default()
                    .model(&model) // Use cloned model string
                    .input(inputs)
                    .build()?; // Propagates OpenAIError

                // eprintln!(
                //     "    Sending request for document {} ({} tokens)... Path: {}",
                //     index + 1,
                //     token_count, // Use correct variable name
                //     doc.path
                // );
                let response = client.embeddings().create(request).await?; // Propagates OpenAIError
                // eprintln!("    Received response for document {}.", index + 1);

                if response.data.len() != 1 {
                    return Err(ServerError::OpenAI(
                        async_openai::error::OpenAIError::ApiError(OpenAIAPIErr {
                            message: format!(
                                "Mismatch in response length for document {}. Expected 1, got {}.",
                                index + 1, response.data.len()
                            ),
                            r#type: Some("sdk_error".to_string()),
                            param: None,
                            code: None,
                        }),
                    ));
                }

                // Process result
                let embedding_data = response.data.first().unwrap(); // Safe unwrap due to check above
                let embedding_array = Array1::from(embedding_data.embedding.clone());
                // Return Ok(Some(...)) for successful embedding, include token count
                Ok(Some((doc.path.clone(), embedding_array, token_count))) // Include token count
            }
        })
        .buffer_unordered(CONCURRENCY_LIMIT) // Run up to CONCURRENCY_LIMIT futures concurrently
        .collect::<Vec<Result<Option<(String, Array1<f32>, usize)>, ServerError>>>() // Update collected result type
        .await;

    // Process collected results, filtering out errors and skipped documents, summing tokens
    let mut embeddings_vec = Vec::new();
    let mut total_processed_tokens: usize = 0;
    for result in results {
        match result {
            Ok(Some((path, embedding, tokens))) => {
                embeddings_vec.push((path, embedding)); // Keep successful embeddings
                total_processed_tokens += tokens; // Add tokens for successful ones
            }
            Ok(None) => {} // Ignore skipped documents
            Err(e) => {
                // Log error but potentially continue? Or return the first error?
                // For now, let's return the first error encountered.
                eprintln!("Error during concurrent embedding generation: {}", e);
                return Err(e);
            }
        }
    }

    eprintln!(
        "Finished generating embeddings. Successfully processed {} documents ({} tokens).",
        embeddings_vec.len(), total_processed_tokens
    );
    Ok((embeddings_vec, total_processed_tokens)) // Return tuple
}