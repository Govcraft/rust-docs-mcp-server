use scraper::{Html, Selector};
use std::{fs::{self, File, create_dir_all}, io::Write}; // Added File, create_dir_all, and Write
use cargo::core::resolver::features::CliFeatures;
// use cargo::core::SourceId; // Removed unused import
// use cargo::util::Filesystem; // Removed unused import

use cargo::core::Workspace;
use cargo::ops::{self, CompileOptions, DocOptions, Packages};
use cargo::util::context::GlobalContext;
use anyhow::Error as AnyhowError;
// use std::process::Command; // Remove Command again
use tempfile::tempdir;
use thiserror::Error;
use walkdir::WalkDir;

#[derive(Debug, Error)]
pub enum DocLoaderError {
    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),
    #[error("WalkDir Error: {0}")]
    WalkDir(#[from] walkdir::Error),
    #[error("CSS selector error: {0}")]
    Selector(String),
    #[error("Temporary directory creation failed: {0}")]
    TempDirCreationFailed(std::io::Error),
    #[error("Cargo library error: {0}")]
    CargoLib(#[from] AnyhowError), // Re-add CargoLib variant
}

// Simple struct to hold document content, maybe add path later if needed
#[derive(Debug, Clone)]
pub struct Document {
    pub path: String,
    pub content: String,
}

/// Generates documentation for a given crate in a temporary directory,
/// then loads and parses the HTML documents.
/// Extracts text content from the main content area of rustdoc generated HTML.
pub fn load_documents(crate_name: &str, crate_version_req: &str) -> Result<Vec<Document>, DocLoaderError> { // Use crate_version_req
    eprintln!("[DEBUG] load_documents called with crate_name: '{}', crate_version_req: '{}'", crate_name, crate_version_req); // Update log
    let mut documents = Vec::new();

    let temp_dir = tempdir().map_err(DocLoaderError::TempDirCreationFailed)?;
    let temp_dir_path = temp_dir.path();
    let temp_manifest_path = temp_dir_path.join("Cargo.toml");

    eprintln!(
        "Generating documentation for crate '{}' (Version Req: '{}') in temporary directory: {}", // Update log message
        crate_name,
        crate_version_req,
        temp_dir_path.display()
    );

    // Create a temporary Cargo.toml using the version requirement
    let cargo_toml_content = format!(
        r#"[package]
name = "temp-doc-crate"
version = "0.1.0"
edition = "2021"

[lib] # Add an empty lib target to satisfy Cargo

[dependencies]
{} = "{}"
"#,
        crate_name, crate_version_req // Use the version requirement string here
    );

    // Create the src directory and an empty lib.rs file
    let src_path = temp_dir_path.join("src");
    create_dir_all(&src_path)?;
    File::create(src_path.join("lib.rs"))?;
    eprintln!("[DEBUG] Created empty src/lib.rs at: {}", src_path.join("lib.rs").display());

    let mut temp_manifest_file = File::create(&temp_manifest_path)?;
    temp_manifest_file.write_all(cargo_toml_content.as_bytes())?;
    eprintln!("[DEBUG] Created temporary manifest at: {}", temp_manifest_path.display());


    // --- Use Cargo API ---
    let mut config = GlobalContext::default()?; // Make mutable
    // Configure context for quiet operation
    config.configure(
        0,     // verbose
        true,  // quiet
        None,  // color
        false, // frozen
        false, // locked
        false, // offline
        &None, // target_dir (Using ws.set_target_dir instead)
        &[],   // unstable_flags
        &[],   // cli_config
    )?;
    // config.shell().set_verbosity(Verbosity::Quiet); // Keep commented

    // Use the temporary manifest path for the Workspace
    let mut ws = Workspace::new(&temp_manifest_path, &config)?; // Make ws mutable
    eprintln!("[DEBUG] Workspace target dir before set: {}", ws.target_dir().as_path_unlocked().display());
    // Set target_dir directly on Workspace
    ws.set_target_dir(cargo::util::Filesystem::new(temp_dir_path.to_path_buf()));
    eprintln!("[DEBUG] Workspace target dir after set: {}", ws.target_dir().as_path_unlocked().display());

    // Create CompileOptions, relying on ::new for BuildConfig
    let mut compile_opts = CompileOptions::new(&config, cargo::core::compiler::CompileMode::Doc { deps: false, json: false })?;
    // Specify the package explicitly
    let package_spec = crate_name.to_string(); // Just use name (with underscores)
    compile_opts.cli_features = CliFeatures::new_all(false); // Use new_all(false)
    compile_opts.spec = Packages::Packages(vec![package_spec.clone()]); // Clone spec

    // Create DocOptions: Pass compile options
    let doc_opts = DocOptions {
        compile_opts,
        open_result: false, // Don't open in browser
        output_format: ops::OutputFormat::Html,
    };
    eprintln!("[DEBUG] package_spec for CompileOptions: '{}'", package_spec);

    ops::doc(&ws, &doc_opts).map_err(DocLoaderError::CargoLib)?; // Use ws
    // --- End Cargo API ---
    // Construct the path to the generated documentation within the temp directory
    // Cargo uses underscores in the directory path if the crate name has hyphens
    let crate_name_underscores = crate_name.replace('-', "_");
    let docs_path = temp_dir_path.join("doc").join(&crate_name_underscores);

    // Debug print relevant options before calling ops::doc
    eprintln!("[DEBUG] CompileOptions spec: {:?}", doc_opts.compile_opts.spec);
    eprintln!("[DEBUG] CompileOptions cli_features: {:?}", doc_opts.compile_opts.cli_features);
    eprintln!("[DEBUG] CompileOptions build_config mode: {:?}", doc_opts.compile_opts.build_config.mode);
    eprintln!("[DEBUG] DocOptions output_format: {:?}", doc_opts.output_format);
    if !docs_path.exists() || !docs_path.is_dir() {
         return Err(DocLoaderError::CargoLib(anyhow::anyhow!(
             "Generated documentation not found at expected path: {}. Check crate name and cargo doc output.",
             docs_path.display()
         )));
    }
    eprintln!("Generated documentation path: {}", docs_path.display());

    eprintln!("[DEBUG] ops::doc called successfully.");

    // Define the CSS selector for the main content area in rustdoc HTML
    // This might need adjustment based on the exact rustdoc version/theme
    let content_selector = Selector::parse("section#main-content.content")
        .map_err(|e| DocLoaderError::Selector(e.to_string()))?;
    eprintln!("[DEBUG] Calculated final docs_path: {}", docs_path.display());

    eprintln!("Starting document loading from: {}", docs_path.display());
        eprintln!("[DEBUG] docs_path does not exist or is not a directory.");

    for entry in WalkDir::new(&docs_path)
        .into_iter()
        .filter_map(Result::ok) // Ignore errors during iteration for now
        .filter(|e| !e.file_type().is_dir() && e.path().extension().is_some_and(|ext| ext == "html"))
    {
        let path = entry.path();
        // Calculate path relative to the docs_path root
        let relative_path = path.strip_prefix(&docs_path).map_err(|e| {
            // Provide more context in the error message
            DocLoaderError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to strip prefix '{}' from path '{}': {}", docs_path.display(), path.display(), e)
            ))
        })?;
        let path_str = relative_path.to_string_lossy().to_string(); // Use the relative path
        // eprintln!("Processing file: {} (relative: {})", path.display(), path_str); // Updated debug log

        // eprintln!("  Reading file content..."); // Added
        let html_content = fs::read_to_string(path)?; // Still read from the absolute path
        // eprintln!("  Parsing HTML..."); // Added

        // Parse the HTML document
        let document = Html::parse_document(&html_content);

        // Select the main content element
        if let Some(main_content_element) = document.select(&content_selector).next() {
            // Extract all text nodes within the main content
            // eprintln!("  Extracting text content..."); // Added
            let text_content: String = main_content_element
                .text()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect::<Vec<&str>>()
                .join("\n"); // Join text nodes with newlines

            if !text_content.is_empty() {
                // eprintln!("  Extracted content ({} chars)", text_content.len()); // Uncommented and simplified
                documents.push(Document {
                    path: path_str,
                    content: text_content,
                });
            } else {
                // eprintln!("No text content found in main section for: {}", path.display()); // Verbose logging
            }
        } else {
             // eprintln!("'main-content' selector not found for: {}", path.display()); // Verbose logging
             // Optionally handle files without the main content selector differently
        }
    }

    eprintln!("Finished document loading. Found {} documents.", documents.len());
    Ok(documents)
}