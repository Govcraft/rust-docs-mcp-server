use rmcp::ServiceError;
use thiserror::Error;
use crate::doc_loader::DocLoaderError;

#[derive(Debug, Error)]
pub enum ServerError {
    #[error("Environment variable not set: {0}")]
    MissingEnvVar(String),
    #[error("Configuration Error: {0}")]
    Config(String),
    #[error("MCP Service Error: {0}")]
    Mcp(#[from] ServiceError),
    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Document Loading Error: {0}")]
    DocLoader(#[from] DocLoaderError),
    #[error("OpenAI Error: {0}")]
    OpenAI(#[from] async_openai::error::OpenAIError),
    #[error("Ollama Error: {0}")]
    Ollama(#[from] ollama_rs::error::OllamaError), // Add Ollama error handling
    #[error("JSON Error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Tiktoken Error: {0}")]
    Tiktoken(String),
    #[error("XDG Directory Error: {0}")]
    Xdg(String),
    #[error("MCP Runtime Error: {0}")]
    McpRuntime(String),
}