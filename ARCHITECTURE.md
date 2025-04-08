# Rust Docs MCP Server Architecture

This document provides an overview of the project's architecture, motivation, and the purpose of its key files.

## Project Overview

### Motivation

AI coding assistants often struggle with the latest APIs of rapidly evolving libraries, particularly in ecosystems like Rust. Their training data has cutoffs, leading to outdated or incorrect suggestions. This project aims to bridge this gap by providing a dedicated, up-to-date knowledge source for specific Rust crates.

### Purpose

The `rustdocs-mcp-server` acts as a specialized Model Context Protocol (MCP) server. Each instance focuses on a single Rust crate (optionally with specific features enabled). It exposes an MCP tool (`query_rust_docs`) that allows an LLM-based coding assistant (like Roo Code, Cursor, etc.) to ask natural language questions about the crate's API or usage. The server retrieves relevant information directly from the crate's *current* documentation and uses another LLM call to synthesize an answer based *only* on that context, significantly improving the accuracy and relevance of the assistant's code generation related to that crate.

### Architecture and Workflow

The application follows these steps:

1.  **Initialization (`src/main.rs`):**
    *   The server is launched via the command line, parsing arguments (`clap`) for the target crate's Package ID Specification (e.g., `serde@^1.0`) and optional features (`-F feat1,feat2`).
    *   It determines the appropriate cache directory (XDG on Linux/macOS, standard cache dir on Windows) based on the crate name, version requirement, and a hash of the requested features.
2.  **Cache Check (`src/main.rs`):**
    *   It checks if pre-computed documentation content and embeddings exist in the cache file (`embeddings.bin`) for the specific crate/version/features combination.
    *   If the cache exists and is valid (`bincode` decodable), it loads the data directly.
3.  **Documentation Generation & Processing (Cache Miss - `src/doc_loader.rs`, `src/main.rs`):**
    *   If the cache is invalid or missing:
        *   A temporary Rust project is created, depending only on the target crate with the specified features enabled in its `Cargo.toml`.
        *   The `cargo` crate's API (`ops::doc`) is used to generate the crate's HTML documentation within the temporary project's `target/doc` directory.
        *   The correct documentation subdirectory (e.g., `target/doc/serde/`) is located by searching for `index.html`.
        *   The generated HTML and Markdown files are walked. `scraper` is used to parse HTML and extract text content from the main content area (`section#main-content.content`). Raw content is used for Markdown files and source code view files (`.rs.html`). Duplicate files (like multiple `index.html` files) are handled, prioritizing the root `index.html` and keeping the largest version of other duplicates.
4.  **Embedding Generation (Cache Miss - `src/embeddings.rs`, `src/main.rs`):**
    *   The extracted text content for each documentation file is sent to the OpenAI API (`async-openai`) to generate vector embeddings using the `text-embedding-3-small` model. Token limits are checked using `tiktoken-rs`.
5.  **Caching (Cache Miss - `src/main.rs`):**
    *   The extracted document content (`Document` struct) and the generated embeddings (`Array1<f32>`) are combined into `CachedDocumentEmbedding` structs and serialized using `bincode` into the `embeddings.bin` cache file in the previously determined path.
6.  **Server Startup (`src/main.rs`, `src/server.rs`):**
    *   An instance of `RustDocsServer` is created, holding the crate name, documents, and embeddings (loaded from cache or freshly generated).
    *   The MCP server is started using the `rmcp` library, listening for connections over standard input/output (stdio).
7.  **Query Handling (`src/server.rs`):**
    *   An MCP client (e.g., Roo Code) connects and calls the `query_rust_docs` tool with a user's question.
    *   The server generates an embedding for the question using OpenAI (`text-embedding-3-small`).
    *   It calculates the cosine similarity between the question embedding and all cached document embeddings to find the most relevant document chunk.
    *   The content of the best-matching document and the original question are sent to the OpenAI chat completion API (`gpt-4o-mini-2024-07-18`).
    *   The LLM is prompted to answer the question based *only* on the provided context.
    *   The LLM's generated answer is sent back to the MCP client as the result of the tool call.
    *   Logging messages are sent asynchronously via MCP notifications throughout the process.

## File Descriptions

### `Cargo.toml`

*   **Purpose:** Defines the Rust package metadata, dependencies, and build configurations.
*   **Key Contents:**
    *   Package name (`rustdocs_mcp_server`), version, edition.
    *   Core Dependencies:
        *   `rmcp`: For MCP server implementation, communication, and macros.
        *   `tokio`: Asynchronous runtime.
        *   `clap`: Command-line argument parsing.
        *   `async-openai`: Interacting with OpenAI APIs (embeddings, chat).
        *   `cargo`: Programmatic interaction with Cargo (specifically `cargo doc`).
        *   `scraper`: HTML parsing for content extraction.
        *   `ndarray`: Vector operations (for embeddings).
        *   `bincode`: Binary serialization/deserialization for caching.
        *   `tiktoken-rs`: Token counting for OpenAI models.
        *   `serde`, `serde_json`: Data serialization/deserialization.
        *   `thiserror`: Error handling boilerplate.
        *   `tempfile`: Creating temporary directories for `cargo doc`.
        *   `walkdir`: Traversing documentation directories.
        *   `xdg` (Linux/macOS), `dirs` (Windows): Locating appropriate cache directories.
        *   `dotenvy`: Loading `.env` files.
    *   Build Profiles: Optimized release profile (`opt-level = "z"`, LTO, strip) for smaller binary size.

### `README.md`

*   **Purpose:** Provides a user-facing overview of the project, including its motivation, features, installation instructions, usage examples, client configuration guidance (Roo Code, Claude Desktop), caching details, and a high-level explanation of how it works.
*   **Key Contents:**
    *   Motivation: Addressing LLM limitations with Rust crate APIs.
    *   Features: Targeted docs, feature support, semantic search, LLM summarization, caching, MCP integration.
    *   Installation: Recommends pre-compiled binaries, provides build-from-source steps.
    *   Usage: Explains how to run the server, the importance of the Package ID Spec, feature flags (`-F`), and the initial run behavior (caching).
    *   MCP Interaction: Details the `query_rust_docs` tool (schema, example) and the `crate://<crate_name>` resource.
    *   Client Configuration Examples: Snippets for `mcp_settings.json` (Roo Code) and Claude Desktop settings.
    *   Caching Explanation: Location (XDG/Windows), format (`bincode`), regeneration logic.
    *   How it Works: Step-by-step description of the server's process flow.
    *   License (MIT) and Sponsorship information.

### `CHANGELOG.md`

*   **Purpose:** Tracks the history of changes, features, and bug fixes across different versions of the project.
*   **Key Contents:** Version numbers, release dates, and categorized lists of changes (Features, Bug Fixes, Code Refactoring, etc.) for each version. Useful for understanding project evolution but not directly involved in runtime execution.

### `src/main.rs`

*   **Purpose:** Serves as the main entry point for the application. Orchestrates the overall process of argument parsing, cache handling, documentation/embedding generation (if needed), and server initialization/startup.
*   **Key Functionality:**
    *   Uses `clap` to parse command-line arguments (package spec, features).
    *   Parses the `PackageIdSpec` to extract crate name and version requirement.
    *   Determines the platform-specific cache directory path using `xdg` or `dirs`, incorporating crate name, version, and a hash of features.
    *   Attempts to load cached `CachedDocumentEmbedding` data using `bincode`.
    *   If cache loading fails or the file doesn't exist:
        *   Calls `doc_loader::load_documents` to generate docs and extract content.
        *   Calls `embeddings::generate_embeddings` to get embeddings from OpenAI.
        *   Calculates estimated OpenAI cost.
        *   Saves the combined documents and embeddings to the cache file using `bincode`.
    *   Initializes the global `OPENAI_CLIENT` static variable.
    *   Instantiates the `RustDocsServer`.
    *   Starts the MCP server using `rmcp::ServiceExt::serve` with the `stdio` transport.
    *   Waits for the server to complete using `server_handle.waiting()`.

### `src/doc_loader.rs`

*   **Purpose:** Handles the generation and processing of Rust documentation for a specified crate.
*   **Key Functionality:**
    *   `load_documents`:
        *   Creates a temporary directory (`tempfile`).
        *   Generates a temporary `Cargo.toml` specifying the target crate and requested features.
        *   Creates a minimal `src/lib.rs` to satisfy Cargo.
        *   Uses the `cargo` crate's API (`ops::doc`, `CompileOptions`, `DocOptions`, `Workspace`) to run `cargo doc` within the temporary environment, targeting only the specified crate.
        *   Calls `find_documentation_path` to locate the correct output directory within `target/doc`.
        *   Calls `process_documentation_directory` to extract content.
    *   `find_documentation_path`: Locates the specific subdirectory within the base `doc` path that contains the generated documentation (handles cases with multiple directories, hyphens/underscores, disambiguates using `index.html` title tag).
    *   `process_documentation_directory`:
        *   Walks the located documentation directory (`walkdir`).
        *   Filters for `.html` and `.md` files.
        *   Groups files by basename to handle duplicates (e.g., multiple `struct.MyStruct.html`).
        *   Filters duplicates, keeping the root `index.html`/`README.md` and the largest HTML file among other duplicates.
        *   For each final file path:
            *   Reads the file content.
            *   If Markdown (`.md`) or source view (`.rs.html`), uses the raw content.
            *   If other HTML, uses `scraper` to parse the HTML and extract text from the `section#main-content.content` element.
        *   Returns a `Vec<Document>` containing the relative path and extracted/raw content.
    *   Defines `Document` struct (path, content) and `DocLoaderError` enum.

### `src/embeddings.rs`

*   **Purpose:** Manages interactions with the OpenAI Embeddings API and provides embedding-related utilities.
*   **Key Functionality:**
    *   Defines `CachedDocumentEmbedding` struct for serialization (includes path, content, vector).
    *   `generate_embeddings`:
        *   Takes a slice of `Document` structs.
        *   Uses `futures::stream` for concurrent requests to the OpenAI API (up to `CONCURRENCY_LIMIT`).
        *   Uses `tiktoken-rs` (`cl100k_base`) to check token count against `TOKEN_LIMIT` before sending requests, skipping documents that exceed the limit.
        *   Calls the OpenAI embeddings endpoint (`text-embedding-3-small`) via `async-openai`.
        *   Collects results, handling potential errors and skipped documents.
        *   Returns a vector of `(String, Array1<f32>)` tuples (path, embedding vector) and the total number of tokens processed.
    *   `cosine_similarity`: Calculates the cosine similarity between two `ndarray::ArrayView1<f32>` vectors.
    *   `OPENAI_CLIENT`: A `OnceLock` static variable to hold the initialized `async_openai::Client`.

### `src/error.rs`

*   **Purpose:** Defines a unified error type for the application using `thiserror`.
*   **Key Contents:**
    *   `ServerError` enum: Consolidates potential errors from various sources.
    *   Variants include: `MissingEnvVar`, `Config`, `Mcp` (from `rmcp::ServiceError`), `Io`, `DocLoader` (from `DocLoaderError`), `OpenAI` (from `async_openai::error::OpenAIError`), `Json` (from `serde_json::Error`), `Tiktoken`, `Xdg`, `McpRuntime`.
    *   Uses `#[from]` attributes to automatically convert underlying error types.

### `src/server.rs`

*   **Purpose:** Implements the core MCP server logic and the `query_rust_docs` tool.
*   **Key Functionality:**
    *   Defines `RustDocsServer` struct: Holds shared state (`Arc`s for crate name, documents, embeddings, peer, startup message).
    *   Implements `ServerHandler` trait for `RustDocsServer`:
        *   `get_info`: Provides server capabilities (tools, logging) and instructions.
        *   `list_resources`, `read_resource`: Basic implementation to expose the crate name as a resource (`crate://<crate_name>`).
        *   Placeholder implementations for `list_prompts`, `get_prompt`, `list_resource_templates`.
    *   Uses `rmcp::tool` macro (`#[tool(tool_box)]`, `#[tool(...)]`) to define the `query_rust_docs` tool:
        *   Takes `QueryRustDocsArgs` (question) as input.
        *   Sends the initial startup message via MCP log notification on the first call.
        *   Gets the embedding for the input question using `OPENAI_CLIENT`.
        *   Iterates through cached document embeddings, calculating `cosine_similarity` to find the best match.
        *   Retrieves the content of the best-matching document.
        *   Constructs system and user prompts for the OpenAI chat completion API (`gpt-4o-mini-2024-07-18`), providing the document context and the user's question.
        *   Calls the chat API via `OPENAI_CLIENT`.
        *   Formats the LLM's response and returns it as a successful `CallToolResult` with text content.
    *   `send_log`: Helper method to send logging messages back to the client via MCP `logging/message` notifications.