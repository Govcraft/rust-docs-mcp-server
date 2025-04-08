use rmcp::ServiceError; // Assuming ServiceError is the correct top-level error
use thiserror::Error;
use crate::doc_loader::DocLoaderError; // Need to import DocLoaderError from the sibling module

#[derive(Debug, Error)]
pub enum ServerError {
    #[error("Environment variable not set: {0}")]
    MissingEnvVar(String),
    // MissingArgument removed as clap handles this now
    #[error("Configuration Error: {0}")]
    Config(String),

    #[error("MCP Service Error: {0}")]
    Mcp(#[from] ServiceError), // Use ServiceError
    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Document Loading Error: {0}")]
    DocLoader(#[from] DocLoaderError),
    #[error("OpenAI Error: {0}")]
    OpenAI(#[from] async_openai::error::OpenAIError),
    #[error("JSON Error: {0}")]
    Json(#[from] serde_json::Error), // Add error for JSON deserialization
    #[error("Tiktoken Error: {0}")]
    Tiktoken(String), // Revert to String, no #[from]
    #[error("XDG Directory Error: {0}")]
    Xdg(String),
    #[error("MCP Runtime Error: {0}")]
    McpRuntime(String),
}