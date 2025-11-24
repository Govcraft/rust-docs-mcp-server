# Rust Docs MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

⭐ **Like this project? Please
[star the repository](https://github.com/Govcraft/rust-docs-mcp-server) on
GitHub to show your support and stay updated!** ⭐

## Motivation

Modern AI-powered coding assistants (like Cursor, Cline, Roo Code, etc.) excel
at understanding code structure and syntax but often struggle with the specifics
of rapidly evolving libraries and frameworks, especially in ecosystems like Rust
where crates are updated frequently. Their training data cutoff means they may
lack knowledge of the latest APIs, leading to incorrect or outdated code
suggestions.

This MCP server addresses this challenge by providing a focused, up-to-date
knowledge source for a specific Rust crate. By running an instance of this
server for a crate (e.g., `serde`, `tokio`, `reqwest`), you give your LLM coding
assistant a tool (`query_rust_docs`) it can use _before_ writing code related to
that crate.

When instructed to use this tool, the LLM can ask specific questions about the
crate's API or usage and receive answers derived directly from the _current_
documentation. This significantly improves the accuracy and relevance of the
generated code, reducing the need for manual correction and speeding up
development.

Multiple instances of this server can be run concurrently, allowing the LLM
assistant to access documentation for several different crates during a coding
session.

This server fetches the documentation for a specified Rust crate, generates
embeddings for the content, and provides an MCP tool to answer questions about
the crate based on the documentation context.

## Features

- **Targeted Documentation:** Focuses on a single Rust crate per server
  instance.
- **Feature Support:** Allows specifying required crate features for
  documentation generation.
- **Semantic Search:** Uses OpenAI's `text-embedding-3-small` model to find the
  most relevant documentation sections for a given question.
- **LLM Summarization:** Leverages OpenAI's `gpt-4o-mini-2024-07-18` model to
  generate concise answers based _only_ on the retrieved documentation context.
- **Caching:** Caches generated documentation content and embeddings in the
  user's XDG data directory (`~/.local/share/rustdocs-mcp-server/` or similar)
  based on crate, version, _and_ requested features to speed up subsequent
  launches.
- **MCP Integration:** Runs as a standard MCP server over stdio, exposing tools
  and resources.

## Prerequisites

- **OpenAI API Key:** Needed for generating embeddings and summarizing answers.
  The server expects this key to be available in the `OPENAI_API_KEY`
  environment variable. (The server also requires network access to download
  crate dependencies and interact with the OpenAI API).

## Installation

The recommended way to install is to download the pre-compiled binary for your
operating system from the
[GitHub Releases page](https://github.com/Govcraft/rust-docs-mcp-server/releases).

1. Go to the
   [Releases page](https://github.com/Govcraft/rust-docs-mcp-server/releases).
2. Download the appropriate archive (`.zip` for Windows, `.tar.gz` for
   Linux/macOS) for your system.
3. Extract the `rustdocs_mcp_server` (or `rustdocs_mcp_server.exe`) binary.
4. Place the binary in a directory included in your system's `PATH` environment
   variable (e.g., `/usr/local/bin`, `~/bin`).

### Building from Source (Alternative)

If you prefer to build from source, you will need the
[Rust Toolchain](https://rustup.rs/) installed.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Govcraft/rust-docs-mcp-server.git
   cd rust-docs-mcp-server
   ```
2. **Build the server:**
   ```bash
   cargo build --release
   ```

## Usage

**Important Note for New Crates:**

When using the server with a crate for the first time (or with a new version/feature set), it needs to download the documentation and generate embeddings. This process can take some time, especially for crates with extensive documentation, and requires an active internet connection and OpenAI API key.

It is recommended to run the server once directly from your command line for any new crate configuration *before* adding it to your AI coding assistant (like Roo Code, Cursor, etc.). This allows the initial embedding generation and caching to complete. Once you see the server startup messages indicating it's ready (e.g., "MCP Server listening on stdio"), you can shut it down (Ctrl+C). Subsequent launches, including those initiated by your coding assistant, will use the cached data and start much faster.


### Running the Server

The server is launched from the command line and requires the **Package ID
Specification** for the target crate. This specification follows the format used
by Cargo (e.g., `crate_name`, `crate_name@version_req`). For the full
specification details, see `man cargo-pkgid` or the
[Cargo documentation](https://doc.rust-lang.org/cargo/reference/pkgid-spec.html).

Optionally, you can specify required crate features using the `-F` or
`--features` flag, followed by a comma-separated list of features. This is
necessary for crates that require specific features to be enabled for
`cargo doc` to succeed (e.g., crates requiring a runtime feature like
`async-stripe`).

```bash
# Set the API key (replace with your actual key)
export OPENAI_API_KEY="sk-..."

# Example: Run server for the latest 1.x version of serde
rustdocs_mcp_server "serde@^1.0"

# Example: Run server for a specific version of reqwest
rustdocs_mcp_server "reqwest@0.12.0"

# Example: Run server for the latest version of tokio
rustdocs_mcp_server tokio

# Example: Run server for async-stripe, enabling a required runtime feature
rustdocs_mcp_server "async-stripe@0.40" -F runtime-tokio-hyper-rustls

# Example: Run server for another crate with multiple features
rustdocs_mcp_server "some-crate@1.2" --features feat1,feat2
```

On the first run for a specific crate version _and feature set_, the server
will:

1. Download the crate documentation using `cargo doc` (with specified features).
2. Parse the HTML documentation.
3. Generate embeddings for the documentation content using the OpenAI API (this
   may take some time and incur costs, though typically only fractions of a US
   penny for most crates; even a large crate like `async-stripe` with over 5000
   documentation pages cost only $0.18 USD for embedding generation during
   testing).
4. Cache the documentation content and embeddings so that the cost isn't
   incurred again.
5. Start the MCP server.

Subsequent runs for the same crate version _and feature set_ will load the data
from the cache, making startup much faster.

### MCP Interaction

The server communicates using the Model Context Protocol over standard
input/output (stdio). It exposes the following:

- **Tool: `query_rust_docs`**
  - **Description:** Query documentation for the specific Rust crate the server
    was started for, using semantic search and LLM summarization.
  - **Input Schema:**
    ```json
    {
      "type": "object",
      "properties": {
        "question": {
          "type": "string",
          "description": "The specific question about the crate's API or usage."
        }
      },
      "required": ["question"]
    }
    ```
  - **Output:** A text response containing the answer generated by the LLM based
    on the relevant documentation context, prefixed with
    `From <crate_name> docs:`.
  - **Example MCP Call:**
    ```json
    {
      "jsonrpc": "2.0",
      "method": "callTool",
      "params": {
        "tool_name": "query_rust_docs",
        "arguments": {
          "question": "How do I make a simple GET request with reqwest?"
        }
      },
      "id": 1
    }
    ```

- **Resource: `crate://<crate_name>`**
  - **Description:** Provides the name of the Rust crate this server instance is
    configured for.
  - **URI:** `crate://<crate_name>` (e.g., `crate://serde`, `crate://reqwest`)
  - **Content:** Plain text containing the crate name.

- **Logging:** The server sends informational logs (startup messages, query
  processing steps) back to the MCP client via `logging/message` notifications.

### Example Client Configuration (Roo Code)

You can configure MCP clients like Roo Code to run multiple instances of this
server, each targeting a different crate. Here's an example snippet for Roo
Code's `mcp_settings.json` file, configuring servers for `reqwest` and
`async-stripe` (note the added features argument for `async-stripe`):

```json
{
  "mcpServers": {
    "rust-docs-reqwest": {
      "command": "/path/to/your/rustdocs_mcp_server",
      "args": [
        "reqwest@0.12"
      ],
      "env": {
        "OPENAI_API_KEY": "YOUR_OPENAI_API_KEY_HERE"
      },
      "disabled": false,
      "alwaysAllow": []
    },
    "rust-docs-async-stripe": {
      "command": "rustdocs_mcp_server",
      "args": [
        "async-stripe@0.40",
        "-F",
        " runtime-tokio-hyper-rustls"
      ],
      "env": {
        "OPENAI_API_KEY": "YOUR_OPENAI_API_KEY_HERE"
      },
      "disabled": false,
      "alwaysAllow": []
    }
  }
}
```

**Note:**

- Replace `/path/to/your/rustdocs_mcp_server` with the actual path to the
  compiled binary on your system if it isn't in your PATH.
- Replace `YOUR_OPENAI_API_KEY_HERE` with your actual OpenAI API key.
- The keys (`rust-docs-reqwest`, `rust-docs-async-stripe`) are arbitrary names
  you choose to identify the server instances within Roo Code.

### Example Client Configuration (Claude Desktop)

For Claude Desktop users, you can configure the server in the MCP settings.
Here's an example configuring servers for `serde` and `async-stripe`:

```json
{
  "mcpServers": {
    "rust-docs-serde": {
      "command": "/path/to/your/rustdocs_mcp_server",
      "args": [
        "serde@^1.0"
      ]
    },
    "rust-docs-async-stripe-rt": {
      "command": "rustdocs_mcp_server",
      "args": [
        "async-stripe@0.40",
        "-F",
        "runtime-tokio-hyper-rustls"
      ]
    }
  }
}
```

**Note:**

- Ensure `rustdocs_mcp_server` is in your system's PATH or provide the full path
  (e.g., `/path/to/your/rustdocs_mcp_server`).
- The keys (`rust-docs-serde`, `rust-docs-async-stripe-rt`) are arbitrary names
  you choose to identify the server instances.
- Remember to set the `OPENAI_API_KEY` environment variable where Claude Desktop
  can access it (this might be system-wide or via how you launch Claude
  Desktop). Claude Desktop's MCP configuration might not directly support
  setting environment variables per-server like Roo Code.
- The example shows how to add the `-F` argument for crates like `async-stripe`
  that require specific features.

### Caching

- **Location:** Cached documentation and embeddings are stored in the XDG data
  directory, typically under
  `~/.local/share/rustdocs-mcp-server/<crate_name>/<sanitized_version_req>/<features_hash>/embeddings.bin`.
  The `sanitized_version_req` is derived from the version requirement, and
  `features_hash` is a hash representing the specific combination of features
  requested at startup. This ensures different feature sets are cached
  separately.
- **Format:** Data is cached using `bincode` serialization.
- **Regeneration:** If the cache file is missing, corrupted, or cannot be
  decoded, the server will automatically regenerate the documentation and
  embeddings.

## How it Works

1. **Initialization:** Parses the crate specification and optional features from
   the command line using `clap`.
2. **Cache Check:** Looks for a pre-existing cache file for the specific crate,
   version requirement, and feature set.
3. **Documentation Generation (if cache miss):**
   - Creates a temporary Rust project depending only on the target crate,
     enabling the specified features in its `Cargo.toml`.
   - Runs `cargo doc` using the `cargo` library API to generate HTML
     documentation in the temporary directory.
   - Dynamically locates the correct output directory within `target/doc` by
     searching for the subdirectory containing `index.html`.
4. **Content Extraction (if cache miss):**
   - Walks the generated HTML files within the located documentation directory.
   - Uses the `scraper` crate to parse each HTML file and extract text content
     from the main content area (`<section id="main-content">`).
5. **Embedding Generation (if cache miss):**
   - Uses the `async-openai` crate and `tiktoken-rs` to generate embeddings for
     each extracted document chunk using the `text-embedding-3-small` model.
   - Calculates the estimated cost based on the number of tokens processed.
6. **Caching (if cache miss):** Saves the extracted document content and their
   corresponding embeddings to the cache file (path includes features hash)
   using `bincode`.
7. **Server Startup:** Initializes the `RustDocsServer` with the
   loaded/generated documents and embeddings.
8. **MCP Serving:** Starts the MCP server using `rmcp` over stdio.
9. **Query Handling (`query_rust_docs` tool):**
   - Generates an embedding for the user's question.
   - Calculates the cosine similarity between the question embedding and all
     cached document embeddings.
   - Identifies the document chunk with the highest similarity.
   - Sends the user's question and the content of the best-matching document
     chunk to the `gpt-4o-mini-2024-07-18` model via the OpenAI API.
   - The LLM is prompted to answer the question based _only_ on the provided
     context.
   - Returns the LLM's response to the MCP client.

## License

This project is licensed under the MIT License.

Copyright (c) 2025 Govcraft

## Sponsor

Govcraft is a one-person shop—no corporate backing, no investors, just me building useful tools. If this project helps you, [sponsoring](https://github.com/sponsors/Govcraft) keeps the work going.

[![Sponsor on GitHub](https://img.shields.io/badge/Sponsor-%E2%9D%A4-%23db61a2?logo=GitHub)](https://github.com/sponsors/Govcraft)
