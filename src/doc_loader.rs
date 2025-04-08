use scraper::{Html, Selector}; // Ensure Html is imported

// Removed duplicate Chunk struct and chunk_html/chunk_markdown functions


// Removed duplicate Chunk struct and chunk_html/chunk_markdown functions


// Removed duplicate scraper import
use std::{fs::{self, File, create_dir_all}, io::{Write}, path::{Path, PathBuf}, ffi::OsStr};
use text_splitter::{ChunkConfig, TextSplitter}; // Import text_splitter
use tiktoken_rs::cl100k_base; // Import tokenizer for chunking
use cargo::core::resolver::features::{CliFeatures};
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
    CargoLib(#[from] AnyhowError),
    #[error("Generic error: {0}")] // Add Generic variant here
    Generic(String),
}

// Simple struct to hold document content, maybe add path later if needed
/// Represents a chunk of extracted documentation content.
#[derive(Debug, Clone)]
pub struct DocumentChunk {
    /// Relative path of the original source file (e.g., "struct.MyStruct.html").
    pub source_path: String,
    /// The actual text content of this chunk.
    pub content: String,
    /// The 0-based index of this chunk within its original source file.
    pub chunk_index: usize,
    // Optional: Could add metadata like section headers later if needed.
}


/// Generates documentation for a given crate in a temporary directory,
/// then loads and parses the HTML and Markdown documents.
/// Extracts text content from the main content area of rustdoc generated HTML,
/// and uses raw content for Markdown files.
pub fn load_documents(
    crate_name: &str,
    crate_version_req: &str,
    features: Option<&Vec<String>>, // Add optional features parameter
) -> Result<Vec<DocumentChunk>, DocLoaderError> { // Return Vec<DocumentChunk>
    // --- Setup Temporary Environment ---
    let temp_dir = tempdir().map_err(DocLoaderError::TempDirCreationFailed)?;
    let temp_dir_path = temp_dir.path();
    let temp_manifest_path = temp_dir_path.join("Cargo.toml");

    // Create a temporary Cargo.toml using the version requirement and features
    let features_string = features
        .filter(|f| !f.is_empty()) // Only add features if provided and not empty
        .map(|f| {
            let feature_list = f.iter().map(|feat| format!("\"{}\"", feat)).collect::<Vec<_>>().join(", ");
            format!(", features = [{}]", feature_list)
        })
        .unwrap_or_default(); // Use empty string if no features

    let cargo_toml_content = format!(
        r#"[package]
name = "temp-doc-crate"
version = "0.1.0"
edition = "2021"

[lib] # Add an empty lib target to satisfy Cargo

[dependencies]
{} = {{ version = "{}"{} }}
"#,
        crate_name, crate_version_req, features_string // Use the version requirement string and features string here
    );

    // Create the src directory and an empty lib.rs file
    let src_path = temp_dir_path.join("src");
    create_dir_all(&src_path)?;
    File::create(src_path.join("lib.rs"))?;

    let mut temp_manifest_file = File::create(&temp_manifest_path)?;
    temp_manifest_file.write_all(cargo_toml_content.as_bytes())?;


    // --- Use Cargo API to Generate Docs ---
    let mut config = GlobalContext::default()?; // Make mutable
    // Configure context (set quiet to false for more detailed errors)
    config.configure(
        0,     // verbose
        true, // quiet
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
    // Set target_dir directly on Workspace
    ws.set_target_dir(cargo::util::Filesystem::new(temp_dir_path.to_path_buf()));

    // Create CompileOptions, relying on ::new for BuildConfig
    let mut compile_opts = CompileOptions::new(&config, cargo::core::compiler::CompileMode::Doc { deps: false, json: false })?;
    // Specify the package explicitly
    let package_spec = crate_name.to_string(); // Just use name (with underscores)
    compile_opts.cli_features = CliFeatures::new_all(false); // Use new_all(false) - applies to the temp crate, not dependency
    compile_opts.spec = Packages::Packages(vec![package_spec.clone()]); // Clone spec

    // Create DocOptions: Pass compile options
    let doc_opts = DocOptions {
        compile_opts,
        open_result: false, // Don't open in browser
        output_format: ops::OutputFormat::Html,
    };


    ops::doc(&ws, &doc_opts).map_err(DocLoaderError::CargoLib)?; // Use ws
    // --- End Cargo API ---

    // --- Find and Process Generated Docs ---
    let base_doc_path = temp_dir_path.join("doc");
    let docs_path = find_documentation_path(&base_doc_path, crate_name)?;

    eprintln!("Using documentation path: {}", docs_path.display()); // Log the path we are actually using

    // Call the refactored processing function
    process_documentation_directory(&docs_path)
}

/// Processes files within a documentation directory, extracting content from HTML and MD files.
fn process_documentation_directory(docs_path: &Path) -> Result<Vec<DocumentChunk>, DocLoaderError> { // Return Vec<DocumentChunk>
    let mut document_chunks = Vec::new(); // Renamed from documents

    // --- Initialize Text Splitter ---
    // Use tiktoken_rs tokenizer for chunk size calculation
    let tokenizer = cl100k_base().map_err(|e| DocLoaderError::Generic(format!("Failed to load tokenizer: {}", e)))?; // Handle potential error
    // Configure chunking: aim for ~500 tokens, trim whitespace. Overlap can be added later if needed.
    let chunk_config = ChunkConfig::new(500) // Target chunk size in tokens
        .with_trim(true);
        // .with_overlap(50) // Optional: Add overlap later if needed

    // Create the splitter using the tokenizer and config
    // Use with_sizer directly with the tiktoken tokenizer (CoreBPE should impl ChunkSizer via feature flag)
    let splitter = TextSplitter::new(chunk_config.with_sizer(tokenizer));
    // Define the CSS selector for the main content area in rustdoc HTML
    let content_selector = Selector::parse("section#main-content.content")
        .map_err(|e| DocLoaderError::Selector(e.to_string()))?;

    // --- Collect all relevant HTML and MD file paths first ---
    let mut relevant_files: Vec<PathBuf> = WalkDir::new(docs_path)
        .into_iter()
        .filter_map(Result::ok) // Ignore errors during iteration
        .filter(|e| {
            if e.file_type().is_dir() { return false; }
            // Check if the extension is either "html" or "md"
            e.path().extension().map_or(false, |ext| ext == "html" || ext == "md")
        })
        .map(|e| e.into_path()) // Get the PathBuf
        .collect(); // Collect into a Vec<PathBuf>

    eprintln!("[DEBUG] Found {} total HTML/MD files initially.", relevant_files.len()); // Log before sorting

    // Sort the paths to ensure deterministic processing order for cache stability.
    relevant_files.sort();

    // --- Use all relevant files directly ---
    let paths_to_process = relevant_files; // Assign directly, no filtering needed
    
    eprintln!("[DEBUG] Processing {} files/paths.", paths_to_process.len());
    
    
    // --- Process the list of files ---
    for path in paths_to_process {
        // Calculate path relative to the docs_path root for storing
        let relative_path = match path.strip_prefix(docs_path) {
            Ok(p) => p.to_path_buf(),
            Err(e) => {
                eprintln!("[WARN] Failed to strip prefix {} from {}: {}", docs_path.display(), path.display(), e);
                continue; // Skip if path manipulation fails
            }
        };
        let source_path_str = relative_path.to_string_lossy().to_string();

        let file_content_raw = match fs::read_to_string(&path) { // Read from the absolute path
            Ok(content) => content,
            Err(e) => {
                eprintln!("[WARN] Failed to read file {}: {}", path.display(), e);
                continue; // Skip this file if reading fails
            }
        };

        // Extract content based on file type
        let content_to_chunk = match path.extension().and_then(OsStr::to_str) {
            Some("md") => {
                eprintln!("[INFO] Processing Markdown: {}", source_path_str);
                file_content_raw // Use raw content for Markdown
            }
            Some("html") => {
                if source_path_str.ends_with(".rs.html") {
                    eprintln!("[INFO] Processing Source Code (raw): {}", source_path_str);
                    // Maybe add basic HTML tag stripping later? For now, use raw.
                    file_content_raw
                } else {
                    eprintln!("[INFO] Processing HTML (parsed): {}", source_path_str);
                    let html = Html::parse_document(&file_content_raw);
                    match html.select(&content_selector).next() {
                        Some(content_node) => {
                            // Extract text, join fragments, and clean whitespace
                            content_node.text().collect::<String>()
                        }
                        None => {
                            eprintln!("[WARN] Could not find main content selector in HTML: {}", source_path_str);
                            String::new() // Use empty string if selector fails
                        }
                    }
                }
            }
            _ => {
                eprintln!("[WARN] Skipping file with unhandled extension: {}", source_path_str);
                String::new() // Skip unhandled types
            }
        };

        // Clean whitespace more thoroughly before chunking
        let cleaned_content = content_to_chunk.split_whitespace().collect::<Vec<_>>().join(" ");

        if cleaned_content.is_empty() {
            eprintln!("[DEBUG] Skipping empty content for: {}", source_path_str);
            continue;
        }

        // --- Chunk the extracted content ---
        let chunks = splitter.chunks(&cleaned_content); // Use the configured splitter
        let mut chunk_count_for_file = 0; // Track chunks per file for logging

        for (index, chunk_text) in chunks.enumerate() {
            if chunk_text.trim().is_empty() {
                continue; // Skip empty chunks after splitting/trimming
            }
            document_chunks.push(DocumentChunk {
                source_path: source_path_str.clone(), // Clone source path for each chunk
                content: chunk_text.to_string(),      // Store the chunk text
                chunk_index: index,                   // Store the 0-based chunk index
            });
            chunk_count_for_file += 1;
        }
         if chunk_count_for_file > 0 {
             eprintln!("[DEBUG] Created {} chunks for: {}", chunk_count_for_file, source_path_str); // Log chunk count only if chunks were created
         }
    }

    eprintln!("Finished document processing. Created {} total chunks.", document_chunks.len()); // Update log message
    Ok(document_chunks) // Return the collected chunks
}

// Removed duplicate DocLoaderError definition (and the stray brace)

/// Finds the correct documentation directory for a specific crate within a base 'doc' directory.
///
/// Handles cases where multiple subdirectories might exist (e.g., due to dependencies)
/// or where the directory name doesn't exactly match the crate name (hyphens vs underscores, renames).
fn find_documentation_path(base_doc_path: &Path, crate_name: &str) -> Result<PathBuf, DocLoaderError> {
    let mut candidate_doc_paths: Vec<PathBuf> = Vec::new();

    if base_doc_path.is_dir() {
        for entry_result in fs::read_dir(base_doc_path)? {
            let entry = entry_result?;
            if entry.file_type()?.is_dir() {
                let dir_path = entry.path();
                // Check if index.html exists within the subdirectory
                if dir_path.join("index.html").is_file() {
                    candidate_doc_paths.push(dir_path);
                }
            }
        }
    }

    match candidate_doc_paths.len() {
        0 => Err(DocLoaderError::CargoLib(anyhow::anyhow!(
            "Could not find any subdirectory containing index.html within '{}'. Cargo doc might have failed or produced unexpected output.",
            base_doc_path.display()
        ))),
        1 => Ok(candidate_doc_paths.remove(0)), // Exactly one candidate, assume it's correct
        _ => {
            // Multiple candidates, try to disambiguate
            let mut matched_path: Option<PathBuf> = None;
            let normalized_input_crate_name = crate_name.replace('-', "_");

            // Prepare scraper selector for title tag
            let title_selector = Selector::parse("html > head > title")
                .map_err(|e| DocLoaderError::Selector(format!("Failed to parse title selector: {}", e)))?;

            for candidate_path in candidate_doc_paths {
                // 1. Check index.html's title tag
                let index_html_path = candidate_path.join("index.html");
                let mut found_match_in_file = false;
                if index_html_path.is_file() {
                    if let Ok(html_content) = fs::read_to_string(&index_html_path) {
                        let document = Html::parse_document(&html_content);
                        if let Some(title_element) = document.select(&title_selector).next() {
                            let title_text = title_element.text().collect::<String>();
                            // Normalize the extracted title part for comparison
                            let normalized_title_crate_part = title_text
                                .split(" - Rust") // Split at " - Rust"
                                .next()          // Take the first part
                                .unwrap_or("")   // Handle cases where " - Rust" isn't present
                                .trim()          // Trim whitespace
                                .replace('-', "_"); // Normalize hyphens

                            if normalized_title_crate_part == normalized_input_crate_name {
                                found_match_in_file = true;
                            }
                        }
                    } else {
                        eprintln!("[WARN] Failed to read index.html at '{}'", index_html_path.display());
                    }
                }

                // 2. If matched via title, track it
                if found_match_in_file {
                    if matched_path.is_some() {
                        // Found a second match! Ambiguous.
                        return Err(DocLoaderError::CargoLib(anyhow::anyhow!(
                            "Found multiple documentation directories matching crate name '{}' based on index.html title within '{}' (e.g., '{}' and '{}'). Cannot determine the correct path.",
                            crate_name, base_doc_path.display(), matched_path.unwrap().display(), candidate_path.display()
                        )));
                    }
                    matched_path = Some(candidate_path);
                }
            }

            // 3. Return the unique match or error
            matched_path.ok_or_else(|| DocLoaderError::CargoLib(anyhow::anyhow!(
                "Found multiple candidate documentation directories within '{}', but none could be uniquely identified as matching crate '{}' using the index.html title.",
                base_doc_path.display(), crate_name


            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;
    use tempfile::tempdir;
    use std::path::Path; // Add Path import

    // Helper to create dummy doc structure including index.html content
    fn setup_test_dir_with_titles(base: &Path, dirs: &[(&str, Option<&str>)]) -> std::io::Result<()> {
        for (name, title_content) in dirs {
            let dir_path = base.join(name);
            fs::create_dir_all(&dir_path)?; // Use create_dir_all

            if let Some(title) = title_content {
                // Create index.html with the specified title
                let index_path = dir_path.join("index.html");
                let mut index_file = File::create(index_path)?;
                // Basic HTML structure with the title
                writeln!(index_file, "<!DOCTYPE html><html><head><title>{}</title></head><body>Content</body></html>", title)?;
            } else {
                // Create an empty index.html if no title specified (or handle differently if needed)
                 File::create(dir_path.join("index.html"))?;
            }

            // Optionally create search-index.js if needed for other tests, but not used for title check
            // let search_index_path = dir_path.join("search-index.js");
            // if fs::metadata(&search_index_path).is_err() { // Avoid overwriting if exists
            //     // Example: Create a dummy search-index.js if required by other logic
            //     // File::create(search_index_path)?;
            // }
        }
        Ok(())
    }

    // Helper to create a mock documentation directory with HTML and MD files
    fn setup_mock_docs(base_path: &Path) -> std::io::Result<()> {
        // Root index.html (should be processed)
        let mut index_file = File::create(base_path.join("index.html"))?;
        writeln!(index_file, "<!DOCTYPE html><html><head><title>Root Crate - Rust</title></head><body><section id='main-content' class='content'>Root Index Content</section></body></html>")?;

        // A regular HTML file (should be processed)
        let mod_path = base_path.join("module");
        fs::create_dir_all(&mod_path)?;
        let mut mod_file = File::create(mod_path.join("struct.MyStruct.html"))?;
        writeln!(mod_file, "<!DOCTYPE html><html><head><title>MyStruct - Rust</title></head><body><section id='main-content' class='content'>MyStruct Content Larger</section></body></html>")?; // Make slightly larger

        // A Markdown file (should be processed, raw content)
        let mut md_file = File::create(base_path.join("README.md"))?;
        writeln!(md_file, "# Project Readme\n\nThis is the content.")?;

        // An HTML file inside a 'src' directory (should be ignored)
        let src_view_path = base_path.join("src").join("my_crate");
        fs::create_dir_all(&src_view_path)?;
        let mut src_html_file = File::create(src_view_path.join("lib.rs.html"))?;
        writeln!(src_html_file, "<html><body>Source Code View</body></html>")?;

        // A duplicate HTML file (only largest should be kept - module one is larger)
        // Create a smaller duplicate in another dir
        let dup_dir = base_path.join("duplicate");
        fs::create_dir_all(&dup_dir)?;
        let mut dup_file = File::create(dup_dir.join("struct.MyStruct.html"))?;
        writeln!(dup_file, "<html><body>Smaller Duplicate Content</body></html>")?; // Smaller content

         // Another Markdown file in a subdirectory
        let mut sub_md_file = File::create(mod_path.join("GUIDE.md"))?;
        writeln!(sub_md_file, "## Guide\n\nMore details here.")?;

        Ok(())
    }


    #[test]
    fn test_find_docs_no_dirs() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let base_path = temp.path();
        let result = find_documentation_path(base_path, "my_crate");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Could not find any subdirectory"));
        Ok(())
    }

    #[test]
    fn test_find_docs_one_dir_correct() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let base_path = temp.path();
        setup_test_dir_with_titles(base_path, &[("my_crate", Some("my_crate - Rust"))])?;
        let result = find_documentation_path(base_path, "my_crate")?;
        assert_eq!(result, base_path.join("my_crate"));
        Ok(())
    }

    #[test]
    fn test_find_docs_one_dir_wrong_name() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let base_path = temp.path();
        setup_test_dir_with_titles(base_path, &[("other_crate", Some("other_crate - Rust"))])?;
        let result = find_documentation_path(base_path, "my_crate")?;
        // If only one dir exists, we assume it's the right one, even if name doesn't match
        assert_eq!(result, base_path.join("other_crate"));
        Ok(())
    }

    #[test]
    fn test_find_docs_multiple_dirs_disambiguate_ok() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let base_path = temp.path();
        setup_test_dir_with_titles(base_path, &[
            ("dep_crate", Some("dep_crate - Rust")),
            ("my_crate", Some("my_crate - Rust")), // Correct title
        ])?;
        let result = find_documentation_path(base_path, "my_crate")?;
        assert_eq!(result, base_path.join("my_crate"));
        Ok(())
    }

    #[test]
    fn test_find_docs_multiple_dirs_hyphen_ok() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let base_path = temp.path();
        setup_test_dir_with_titles(base_path, &[
            ("dep_crate", Some("dep_crate - Rust")),
            // Crate name has hyphen, title might use underscore or hyphen
            ("my_crate_docs", Some("my_crate - Rust")), // Title uses underscore matching normalized name
        ])?;
        let result = find_documentation_path(base_path, "my-crate")?; // Input has hyphen
        assert_eq!(result, base_path.join("my_crate_docs"));
        Ok(())
    }

    #[test]
    fn test_find_docs_multiple_dirs_no_match() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let base_path = temp.path();
        setup_test_dir_with_titles(base_path, &[
            ("dep_crate", Some("dep_crate - Rust")),
            ("another_dep", Some("another_dep - Rust")),
        ])?;
        let result = find_documentation_path(base_path, "my_crate");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("none could be uniquely identified"));
        Ok(())
    }

    #[test]
    fn test_find_docs_multiple_dirs_ambiguous_match() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let base_path = temp.path();
        setup_test_dir_with_titles(base_path, &[
            ("my_crate_v1", Some("my_crate - Rust")), // Matches normalized "my_crate"
            ("my_crate_v2", Some("my_crate - Rust")), // Also matches normalized "my_crate"
        ])?;
        let result = find_documentation_path(base_path, "my_crate");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Found multiple documentation directories matching"));
        Ok(())
    }

    #[test]
    fn test_find_docs_multiple_dirs_missing_index_html() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let base_path = temp.path();
         // Create dirs but only one with index.html
        fs::create_dir_all(base_path.join("dep_crate"))?;
        setup_test_dir_with_titles(base_path, &[("my_crate", Some("my_crate - Rust"))])?;

        let result = find_documentation_path(base_path, "my_crate")?;
        // Should still find the correct one as the other is not a candidate
        assert_eq!(result, base_path.join("my_crate"));
        Ok(())
    }

     #[test]
    fn test_find_docs_multiple_dirs_malformed_title() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let base_path = temp.path();
        setup_test_dir_with_titles(base_path, &[
            ("dep_crate", Some("Completely Wrong Title Format")), // Malformed title
            ("my_crate", Some("my_crate - Rust")),
        ])?;
        let result = find_documentation_path(base_path, "my_crate")?;
        // Should ignore malformed and find correct one
        assert_eq!(result, base_path.join("my_crate"));
        Ok(())
    }

    #[test]
    fn test_find_docs_multiple_dirs_disambiguate_by_title() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let base_path = temp.path();
        // Simulate missing search-index.js but correct title in index.html
        setup_test_dir_with_titles(base_path, &[
            ("stdio_fixture", Some("stdio_fixture - Rust")),
            ("clap", Some("clap - Rust")),
        ])?;

        let result = find_documentation_path(base_path, "clap")?;
        assert_eq!(result, base_path.join("clap"));
        Ok(())
    }

    #[test]
    fn test_process_documentation_directory_includes_md() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let docs_path = temp.path();
        setup_mock_docs(docs_path)?;

        let documents = process_documentation_directory(docs_path)?;

         // Expect 5 documents now: index.html, MyStruct.html, README.md, GUIDE.md, and the source file lib.rs.html
         assert_eq!(documents.len(), 5, "Should find root index.html, MyStruct.html, README.md, GUIDE.md, and lib.rs.html");

         // Check for specific documents (order might vary)
         let mut found_index = false;
         let mut found_struct = false;
         let mut found_readme = false;
         let mut found_source = false; // Added check for source file
         let mut found_guide = false;

         for doc in &documents {
               // eprintln!("Found doc: path='{}', content='{}'", doc.path, doc.content.chars().take(50).collect::<String>()); // Commented out debug print
              if doc.source_path == "index.html" {
                  assert!(doc.content.contains("Root Index Content"));
                  found_index = true;
              } else if doc.source_path == "module/struct.MyStruct.html" { // Path relative to docs_path
                 assert!(doc.content.contains("MyStruct Content Larger")); // Check content of the larger one
                 found_struct = true;
              } else if doc.source_path == "README.md" { // Root README
                  assert!(doc.content.contains("# Project Readme"));
                  assert!(doc.content.contains("This is the content."));
                  found_readme = true;
              } else if doc.source_path == "module/GUIDE.md" { // Path relative to docs_path
                   assert!(doc.content.contains("## Guide"));
                   assert!(doc.content.contains("More details here."));
                   found_guide = true;
              } else if doc.source_path == "src/my_crate/lib.rs.html" { // Check for source file
                   assert!(doc.content.contains("Source Code View")); // Check raw content
                   found_source = true;
              }
         }

         assert!(found_index, "Root index.html content not found or incorrect");
         assert!(found_struct, "MyStruct.html content not found or incorrect");
         assert!(found_readme, "README.md content not found or incorrect");
         assert!(found_source, "Source file (lib.rs.html) not found or incorrect"); // Added assertion
         assert!(found_guide, "module/GUIDE.md content not found or incorrect");

         // Verify ignored files are not present
         // Check that the smaller duplicate wasn't included
         assert!(!documents.iter().any(|d| d.source_path == "duplicate/struct.MyStruct.html"), "Smaller duplicate should be ignored");

         Ok(())
    }

}
