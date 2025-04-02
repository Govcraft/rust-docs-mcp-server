use scraper::{Html, Selector};
use std::{collections::HashMap, fs::{self, File, create_dir_all}, io::{Write}, path::{Path, PathBuf}, ffi::OsStr}; // Added OsStr
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
    CargoLib(#[from] AnyhowError), // Re-add CargoLib variant
    // Removed unused StripPrefix variant
}

// Simple struct to hold document content, maybe add path later if needed
#[derive(Debug, Clone)]
pub struct Document {
    pub path: String,
    pub content: String,
}


/// Generates documentation for a given crate in a temporary directory,
/// then loads and parses the HTML and Markdown documents.
/// Extracts text content from the main content area of rustdoc generated HTML,
/// and uses raw content for Markdown files.
pub fn load_documents(
    crate_name: &str,
    crate_version_req: &str,
    features: Option<&Vec<String>>, // Add optional features parameter
) -> Result<Vec<Document>, DocLoaderError> {
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
fn process_documentation_directory(docs_path: &Path) -> Result<Vec<Document>, DocLoaderError> {
    let mut documents = Vec::new();
    // Define the CSS selector for the main content area in rustdoc HTML
    // This might need adjustment based on the exact rustdoc version/theme
    let content_selector = Selector::parse("section#main-content.content")
        .map_err(|e| DocLoaderError::Selector(e.to_string()))?;

    // --- Collect all relevant HTML and MD file paths first ---
    let relevant_files: Vec<PathBuf> = WalkDir::new(docs_path)
        .into_iter()
        .filter_map(Result::ok) // Ignore errors during iteration
        .filter(|e| {
            if e.file_type().is_dir() { return false; }
            // Check if the extension is either "html" or "md"
            e.path().extension().map_or(false, |ext| ext == "html" || ext == "md")
        })
        .map(|e| e.into_path()) // Get the PathBuf
        .collect();

    eprintln!("[DEBUG] Found {} total HTML/MD files initially.", relevant_files.len());

    // --- Group files by basename to handle duplicates (primarily for HTML) ---
    let mut basename_groups: HashMap<String, Vec<PathBuf>> = HashMap::new();
    for path in relevant_files { // Use the combined list
        if let Some(filename_osstr) = path.file_name() {
            if let Some(filename_str) = filename_osstr.to_str() {
                basename_groups
                    .entry(filename_str.to_string())
                    .or_default()
                    .push(path);
            } else {
                 eprintln!("[WARN] Skipping file with non-UTF8 name: {}", path.display());
            }
        } else {
             eprintln!("[WARN] Skipping file with no name: {}", path.display());
        }
    }

    // --- Initialize paths_to_process and explicitly add the root index.html if it exists ---
    // This ensures the main crate page is always included if present.
    let mut paths_to_process: Vec<PathBuf> = Vec::new();
    let root_index_path = docs_path.join("index.html");
    if root_index_path.is_file() {
        paths_to_process.push(root_index_path);
    }
    // Also check for a root README.md
    let root_readme_path = docs_path.join("README.md");
     if root_readme_path.is_file() && !paths_to_process.contains(&root_readme_path) { // Avoid adding if index.html was README.md (unlikely)
         paths_to_process.push(root_readme_path);
     }


    // --- Filter based on duplicates (keep largest HTML) and ignore source view ---
    for (basename, mut paths) in basename_groups {
        // Always ignore index.html and README.md at this stage, as the root ones were handled above.
        // This prevents including module index pages or nested readmes multiple times if they share names.
        if basename == "index.html" || basename == "README.md" {
            continue;
        }

        // Also ignore files within source code view directories (e.g., `doc/src/...`)
        // Check the first path (they should share the problematic component if any)
        if paths.first().map_or(false, |p| p.components().any(|comp| comp.as_os_str() == OsStr::new("src"))) {
             eprintln!("[DEBUG] Ignoring file in src view: {}", paths.first().unwrap().display());
             continue;
        }


        if paths.len() == 1 {
            // Single file with this basename (and not index.html/README.md), keep it
            paths_to_process.push(paths.remove(0));
        } else {
            // Multiple files with the same basename (likely HTML duplicates)
            // Find the largest one by file size - typically the main definition page vs. re-exports.
            // Explicit type annotation needed for the error type in try_fold
            let largest_path_result: Result<Option<(PathBuf, u64)>, std::io::Error> = paths.into_iter().try_fold(None::<(PathBuf, u64)>, |largest, current| {
                // Only consider HTML files for size comparison duplicate resolution
                if current.extension().map_or(false, |ext| ext != "html") {
                    return Ok(largest); // Skip non-HTML files in this check
                }
                let current_meta = fs::metadata(&current)?;
                let current_size = current_meta.len();
                match largest {
                    None => Ok(Some((current, current_size))),
                    Some((largest_path_so_far, largest_size_so_far)) => {
                        if current_size > largest_size_so_far {
                            Ok(Some((current, current_size)))
                        } else {
                            Ok(Some((largest_path_so_far, largest_size_so_far)))
                        }
                    }
                }
            });

            match largest_path_result {
                Ok(Some((p, _size))) => {
                    // eprintln!("[DEBUG] Duplicate basename '{}': Keeping largest file {}", basename, p.display());
                    paths_to_process.push(p);
                }
                Ok(None) => {
                     // This case might happen if all duplicates were non-HTML
                     eprintln!("[WARN] No HTML files found for basename '{}' during size comparison, or group was empty.", basename);
                }
                Err(e) => {
                    eprintln!("[WARN] Error getting metadata for basename '{}', skipping group: {}", basename, e);
                    // Skip the whole group if metadata fails
                }
            }
        }
    }

     eprintln!("[DEBUG] Filtered down to {} unique files/paths to process.", paths_to_process.len());


    // --- Process the filtered list of files ---
    for path in paths_to_process {
        // Calculate path relative to the docs_path root for storing in Document
        let relative_path = match path.strip_prefix(docs_path) {
             Ok(p) => p.to_path_buf(),
            Err(e) => {
                 eprintln!("[WARN] Failed to strip prefix {} from {}: {}", docs_path.display(), path.display(), e);
                 continue; // Skip if path manipulation fails
             }
        };
        let path_str = relative_path.to_string_lossy().to_string();

        let file_content = match fs::read_to_string(&path) { // Read from the absolute path
            Ok(content) => content,
            Err(e) => {
                eprintln!("[WARN] Failed to read file {}: {}", path.display(), e);
                continue; // Skip this file if reading fails
            }
        };

        // Check file extension to decide processing method
        if path.extension().map_or(false, |ext| ext == "html") {
            // Process HTML using scraper
            let html_document = Html::parse_document(&file_content);
            if let Some(main_content_element) = html_document.select(&content_selector).next() {
                let text_content: String = main_content_element
                    .text()
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<&str>>()
                    .join("\n");

                if !text_content.is_empty() {
                    documents.push(Document {
                        path: path_str,
                        content: text_content,
                    });
                } else {
                     // eprintln!("[DEBUG] No text content found in main section for HTML: {}", path.display());
                }
            } else {
                 // eprintln!("[DEBUG] 'main-content' selector not found for HTML: {}", path.display());
            }
        } else if path.extension().map_or(false, |ext| ext == "md") {
            // Process Markdown: Use raw content
            if !file_content.trim().is_empty() {
                documents.push(Document {
                    path: path_str,
                    content: file_content, // Store the raw Markdown content
                });
            } else {
                 eprintln!("[DEBUG] Skipping empty Markdown file: {}", path.display());
            }
        } else {
            // Should not happen due to WalkDir filter, but handle defensively
            eprintln!("[WARN] Skipping file with unexpected extension: {}", path.display());
        }
    }

    eprintln!("Finished document loading. Found {} final documents.", documents.len());
    Ok(documents)
}


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

        assert_eq!(documents.len(), 4, "Should find root index.html, MyStruct.html, README.md, and GUIDE.md");

        // Check for specific documents (order might vary)
        let mut found_index = false;
        let mut found_struct = false;
        let mut found_readme = false;
        let mut found_guide = false;

        for doc in &documents {
             eprintln!("Found doc: path='{}', content='{}'", doc.path, doc.content.chars().take(50).collect::<String>()); // Debug print
            if doc.path == "index.html" {
                assert!(doc.content.contains("Root Index Content"));
                found_index = true;
            } else if doc.path == "module/struct.MyStruct.html" { // Path relative to docs_path
                assert!(doc.content.contains("MyStruct Content Larger")); // Check content of the larger one
                found_struct = true;
            } else if doc.path == "README.md" {
                assert!(doc.content.contains("# Project Readme"));
                assert!(doc.content.contains("This is the content."));
                found_readme = true;
            } else if doc.path == "module/GUIDE.md" { // Path relative to docs_path
                 assert!(doc.content.contains("## Guide"));
                 assert!(doc.content.contains("More details here."));
                 found_guide = true;
            }
        }

        assert!(found_index, "Root index.html content not found or incorrect");
        assert!(found_struct, "MyStruct.html content not found or incorrect");
        assert!(found_readme, "README.md content not found or incorrect");
        assert!(found_guide, "module/GUIDE.md content not found or incorrect");

        // Verify ignored files are not present
        assert!(!documents.iter().any(|d| d.path.contains("src/")), "Should ignore files in 'src/' directories");
        // Check that the smaller duplicate wasn't included
        assert!(!documents.iter().any(|d| d.path == "duplicate/struct.MyStruct.html"), "Smaller duplicate should be ignored");

        Ok(())
    }

}
