# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Rust-based MCP (Model Context Protocol) server that provides AI coding assistants with access to up-to-date Rust crate documentation. The server fetches documentation for a specified crate, generates embeddings using OpenAI's API, and provides semantic search capabilities through an MCP tool.

## Development Commands

### Build and Run
```bash
# Build the project
cargo build --release

# Run the server (requires OPENAI_API_KEY environment variable)
export OPENAI_API_KEY="sk-..."
cargo run -- "crate_name@version"

# Example: Run server for serde
cargo run -- "serde@^1.0"

# Example: Run with specific features
cargo run -- "async-stripe@0.40" -F runtime-tokio-hyper-rustls
```

### Development Tools
```bash
# Check code formatting
cargo fmt --check

# Run linter
cargo clippy

# Run tests
cargo test

# Build with Nix (if available)
nix build
```

### Environment Setup

The project requires:
- OpenAI API key set as `OPENAI_API_KEY` environment variable
- Internet connection for downloading crate documentation and API calls
- Rust toolchain (edition 2024)

Optional environment variables:
- `EMBEDDING_MODEL`: OpenAI embedding model (default: "text-embedding-3-small")
- `LLM_MODEL`: OpenAI chat model (default: "gpt-4o-mini-2024-07-18")
- `OPENAI_API_BASE`: Custom OpenAI API base URL

## Architecture

### Core Components

1. **main.rs**: Entry point that handles CLI parsing, cache management, and server initialization
2. **server.rs**: Contains `RustDocsServer` implementing MCP `ServerHandler` trait
3. **doc_loader.rs**: Handles downloading and parsing Rust crate documentation using Cargo API
4. **embeddings.rs**: Manages OpenAI API integration for generating and processing embeddings
5. **error.rs**: Centralized error handling

### Key Architecture Patterns

- **Caching Strategy**: Documents and embeddings are cached in XDG data directory (`~/.local/share/rustdocs-mcp-server/`) with paths incorporating crate name, version, and feature hash
- **Async Processing**: Uses concurrent embedding generation with configurable limits (8 concurrent requests, 8000 token limit per document)
- **MCP Integration**: Implements standard MCP protocol over stdio using the `rmcp` crate
- **Dynamic Tool Naming**: Tool names are dynamically generated as `query_{crate_name}_docs` to support multiple server instances

### Data Flow

1. CLI parses crate specification and features
2. Check cache for existing embeddings (keyed by crate + version + features)
3. If cache miss: download docs using Cargo API, generate embeddings via OpenAI, cache results
4. Initialize MCP server with loaded documents and embeddings
5. Handle `query_{crate_name}_docs` tool calls by finding best matching document via cosine similarity
6. Generate responses using OpenAI chat API with matched document as context

### Important Implementation Details

- **Feature Support**: Crate features are properly handled during documentation generation and included in cache keys
- **Document Filtering**: Implements intelligent filtering to avoid duplicate HTML files and source code views
- **Error Handling**: Comprehensive error types covering IO, Cargo API, OpenAI API, and MCP protocol errors
- **Resource Management**: Proper cleanup of temporary directories and concurrent request limiting

## Testing Strategy

When making changes:
1. Test with different crate specifications (name only, version requirements, features)
2. Verify caching behavior by running same crate twice
3. Test MCP protocol compliance with actual MCP clients
4. Validate OpenAI API integration with different embedding/chat models
5. Test error handling for invalid crate names, missing API keys, network issues

## Configuration Files

- `Cargo.toml`: Standard Rust project configuration with optimized release profile
- `flake.nix`: Nix development environment with Rust toolchain and OpenSSL dependencies
- Platform-specific dependency handling for Windows vs Unix systems