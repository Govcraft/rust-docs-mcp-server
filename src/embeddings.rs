use crate::{doc_loader::Document, error::ServerError};
use async_openai::{
    config::OpenAIConfig, error::ApiError as OpenAIAPIErr, types::CreateEmbeddingRequestArgs,
    Client as OpenAIClient,
};
use ndarray::{Array1, ArrayView1};
use ollama_rs::{
    generation::embeddings::request::GenerateEmbeddingsRequest,
    Ollama,
};
use std::sync::OnceLock;
use std::sync::Arc;
use tiktoken_rs::cl100k_base;
use futures::stream::{self, StreamExt};

// Static OnceLocks for both clients
pub static OPENAI_CLIENT: OnceLock<OpenAIClient<OpenAIConfig>> = OnceLock::new();
pub static OLLAMA_CLIENT: OnceLock<Ollama> = OnceLock::new();

use bincode::{Encode, Decode};
use serde::{Serialize, Deserialize};

// Define a struct containing path, content, and embedding for caching
#[derive(Serialize, Deserialize, Debug, Encode, Decode)]
pub struct CachedDocumentEmbedding {
    pub path: String,
    pub content: String,
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

/// Generates embeddings using Ollama with the nomic-embed-text model
pub async fn generate_ollama_embeddings(
    ollama_client: &Ollama,
    documents: &[Document],
    model: &str,
) -> Result<Vec<(String, Array1<f32>)>, ServerError> {
    eprintln!("Generating embeddings for {} documents using Ollama...", documents.len());

    const CONCURRENCY_LIMIT: usize = 4; // Lower concurrency for Ollama
    const TOKEN_LIMIT: usize = 8000; // Adjust based on your model's limits

    // Get the tokenizer (we'll use this for approximate token counting)
    let bpe = Arc::new(cl100k_base().map_err(|e| ServerError::Tiktoken(e.to_string()))?);

    let results = stream::iter(documents.iter().enumerate())
        .map(|(index, doc)| {
            let ollama_client = ollama_client.clone();
            let model = model.to_string();
            let doc = doc.clone();
            let bpe = Arc::clone(&bpe);

            async move {
                // Approximate token count for filtering
                let token_count = bpe.encode_with_special_tokens(&doc.content).len();

                if token_count > TOKEN_LIMIT {
                    eprintln!(
                        "    Skipping document {}: Approximate tokens ({}) exceed limit ({}). Path: {}",
                        index + 1,
                        token_count,
                        TOKEN_LIMIT,
                        doc.path
                    );
                    return Ok::<Option<(String, Array1<f32>)>, ServerError>(None);
                }

                eprintln!(
                    "    Processing document {} (approx {} tokens)... Path: {}",
                    index + 1,
                    token_count,
                    doc.path
                );

                // Create embeddings request for Ollama
                let request = GenerateEmbeddingsRequest::new(
                    model,
                    doc.content.clone().into(),
                );

                match ollama_client.generate_embeddings(request).await {
                    Ok(response) => {
                        if let Some(embedding) = response.embeddings.first() {
                            let embedding_array = Array1::from(embedding.clone());
                            eprintln!("    Received response for document {}.", index + 1);
                            Ok(Some((doc.path.clone(), embedding_array)))
                        } else {
                            Err(ServerError::Config(format!(
                                "No embeddings returned for document {}",
                                index + 1
                            )))
                        }
                    }
                    Err(e) => Err(ServerError::Config(format!(
                        "Ollama embedding error for document {}: {}",
                        index + 1, e
                    )))
                }
            }
        })
        .buffer_unordered(CONCURRENCY_LIMIT)
        .collect::<Vec<Result<Option<(String, Array1<f32>)>, ServerError>>>()
        .await;

    // Process collected results
    let mut embeddings_vec = Vec::new();
    for result in results {
        match result {
            Ok(Some((path, embedding))) => {
                embeddings_vec.push((path, embedding));
            }
            Ok(None) => {} // Skipped document
            Err(e) => {
                eprintln!("Error during Ollama embedding generation: {}", e);
                return Err(e);
            }
        }
    }

    eprintln!(
        "Finished generating Ollama embeddings. Successfully processed {} documents.",
        embeddings_vec.len()
    );
    Ok(embeddings_vec)
}

/// Generates embeddings for a single text using Ollama (for questions)
pub async fn generate_single_ollama_embedding(
    ollama_client: &Ollama,
    text: &str,
    model: &str,
) -> Result<Array1<f32>, ServerError> {
    let request = GenerateEmbeddingsRequest::new(
        model.to_string(),
        text.to_string().into(),
    );

    match ollama_client.generate_embeddings(request).await {
        Ok(response) => {
            if let Some(embedding) = response.embeddings.first() {
                Ok(Array1::from(embedding.clone()))
            } else {
                Err(ServerError::Config("No embedding returned".to_string()))
            }
        }
        Err(e) => Err(ServerError::Config(format!(
            "Ollama embedding error: {}",
            e
        )))
    }
}

/// Legacy OpenAI embedding generation (kept for fallback)
pub async fn generate_openai_embeddings(
    client: &OpenAIClient<OpenAIConfig>,
    documents: &[Document],
    model: &str,
) -> Result<(Vec<(String, Array1<f32>)>, usize), ServerError> {
    // Keep the original OpenAI implementation for fallback
    let bpe = Arc::new(cl100k_base().map_err(|e| ServerError::Tiktoken(e.to_string()))?);

    const CONCURRENCY_LIMIT: usize = 8;
    const TOKEN_LIMIT: usize = 8000;

    let results = stream::iter(documents.iter().enumerate())
        .map(|(index, doc)| {
            let client = client.clone();
            let model = model.to_string();
            let doc = doc.clone();
            let bpe = Arc::clone(&bpe);

            async move {
                let token_count = bpe.encode_with_special_tokens(&doc.content).len();

                if token_count > TOKEN_LIMIT {
                    return Ok::<Option<(String, Array1<f32>, usize)>, ServerError>(None);
                }

                let inputs: Vec<String> = vec![doc.content.clone()];
                let request = CreateEmbeddingRequestArgs::default()
                    .model(&model)
                    .input(inputs)
                    .build()?;

                let response = client.embeddings().create(request).await?;

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

                let embedding_data = response.data.first().unwrap();
                let embedding_array = Array1::from(embedding_data.embedding.clone());
                Ok(Some((doc.path.clone(), embedding_array, token_count)))
            }
        })
        .buffer_unordered(CONCURRENCY_LIMIT)
        .collect::<Vec<Result<Option<(String, Array1<f32>, usize)>, ServerError>>>()
        .await;

    let mut embeddings_vec = Vec::new();
    let mut total_processed_tokens: usize = 0;
    for result in results {
        match result {
            Ok(Some((path, embedding, tokens))) => {
                embeddings_vec.push((path, embedding));
                total_processed_tokens += tokens;
            }
            Ok(None) => {}
            Err(e) => {
                eprintln!("Error during OpenAI embedding generation: {}", e);
                return Err(e);
            }
        }
    }

    eprintln!(
        "Finished generating OpenAI embeddings. Successfully processed {} documents ({} tokens).",
        embeddings_vec.len(), total_processed_tokens
    );
    Ok((embeddings_vec, total_processed_tokens))
}

/// Main embedding generation function that tries Ollama first, falls back to OpenAI
pub async fn generate_embeddings(
    documents: &[Document],
    model: &str,
) -> Result<(Vec<(String, Array1<f32>)>, usize), ServerError> {
    // Check if Ollama is available
    if let Some(ollama_client) = OLLAMA_CLIENT.get() {
        eprintln!("Using Ollama for embedding generation with model: {}", model);
        // For Ollama, we don't track tokens the same way, so return 0 for token count
        let embeddings = generate_ollama_embeddings(ollama_client, documents, model).await?;
        Ok((embeddings, 0))
    } else if let Some(openai_client) = OPENAI_CLIENT.get() {
        eprintln!("Fallback to OpenAI for embedding generation with model: {}", model);
        generate_openai_embeddings(openai_client, documents, model).await
    } else {
        Err(ServerError::Config(
            "No embedding client available (neither Ollama nor OpenAI)".to_string()
        ))
    }
}