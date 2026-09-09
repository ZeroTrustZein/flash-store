use serde::{Deserialize, Serialize};

use crate::rag::types::{DocumentMetadata, ScoredDocument};

/// Format layout for assembled RAG context blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ContextFormat {
    /// Markdown sections with headings and scores:
    /// `### [1] doc_id (Score: 0.95)\n\ntext...`
    #[default]
    Markdown,
    /// XML structured document tags:
    /// `<document id="doc_id" index="1" score="0.95">\ntext...\n</document>`
    Xml,
    /// Numbered citation list:
    /// `[1] [doc_id]: text...`
    Numbered,
    /// Compact single-block format with minimal whitespace.
    Compact,
}

/// Strategy for handling documents that exceed the available token/character budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TruncationStrategy {
    /// If a candidate document does not completely fit within the remaining budget, stop and drop it.
    DropOversized,
    /// Include as many full documents as fit, then truncate the final document's text to fill the remaining budget.
    #[default]
    TruncateLast,
}

/// Configuration options for assembling RAG context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    /// Maximum character budget for the assembled context (defaults to 16,384 chars ~ 4,000 tokens).
    pub max_chars: usize,
    /// Layout format for the document passages.
    pub format: ContextFormat,
    /// How to handle documents when approaching budget limit.
    pub truncation_strategy: TruncationStrategy,
    /// Whether to include the retrieval relevance score in header/tags.
    pub include_scores: bool,
    /// Whether to include key metadata fields (e.g. source, author, url).
    pub include_metadata: bool,
    /// Header text prepended before the documents (e.g. "Context passages:").
    pub header: Option<String>,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_chars: 16_384,
            format: ContextFormat::Markdown,
            truncation_strategy: TruncationStrategy::TruncateLast,
            include_scores: true,
            include_metadata: true,
            header: Some("Relevant Context Passages:".to_string()),
        }
    }
}

impl ContextConfig {
    /// Creates a new `ContextConfig` with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum character limit.
    pub fn max_chars(mut self, chars: usize) -> Self {
        self.max_chars = chars;
        self
    }

    /// Sets the formatting layout.
    pub fn format(mut self, format: ContextFormat) -> Self {
        self.format = format;
        self
    }

    /// Sets the truncation strategy.
    pub fn truncation_strategy(mut self, strategy: TruncationStrategy) -> Self {
        self.truncation_strategy = strategy;
        self
    }

    /// Configures whether relevance scores are included in headers.
    pub fn include_scores(mut self, include: bool) -> Self {
        self.include_scores = include;
        self
    }

    /// Configures whether metadata is included in headers.
    pub fn include_metadata(mut self, include: bool) -> Self {
        self.include_metadata = include;
        self
    }

    /// Sets an optional header title.
    pub fn header(mut self, header: impl Into<String>) -> Self {
        self.header = Some(header.into());
        self
    }

    /// Clears any header text.
    pub fn without_header(mut self) -> Self {
        self.header = None;
        self
    }
}

/// Citation reference metadata associating a prompt index with source document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Citation {
    /// 1-based index appearing in the assembled context prompt (e.g. [1]).
    pub index: usize,
    /// Underlying document identifier.
    pub doc_id: String,
    /// Combined retrieval / reranking score.
    pub score: f32,
    /// Source URI or origin file if present in metadata.
    pub source: Option<String>,
    /// Associated document metadata.
    pub metadata: DocumentMetadata,
}

/// Result of context assembly containing formatted text and source citations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssembledContext {
    /// Final assembled text ready to inject into LLM system/user prompt.
    pub text: String,
    /// List of citations for source attribution.
    pub citations: Vec<Citation>,
    /// Number of documents included (fully or partially).
    pub doc_count: usize,
    /// Total character count of `text`.
    pub total_chars: usize,
    /// Whether any document was truncated due to budget constraints.
    pub was_truncated: bool,
}

/// Context assembly subsystem that transforms retrieved `ScoredDocument`s
/// into formatted LLM prompts with token budgeting, citations, and layout control.
pub struct ContextAssembler;

impl ContextAssembler {
    /// Assembles an array of `ScoredDocument`s into an `AssembledContext`.
    pub fn assemble(documents: &[ScoredDocument], config: &ContextConfig) -> AssembledContext {
        let mut assembled_blocks = Vec::new();
        let mut citations = Vec::new();
        let mut current_chars = 0;
        let mut was_truncated = false;

        let header_text = if let Some(ref h) = config.header {
            let h_str = format!("{}\n\n", h);
            current_chars += h_str.len();
            Some(h_str)
        } else {
            None
        };

        for (idx, doc) in documents.iter().enumerate() {
            let citation_idx = idx + 1;
            let doc_text = doc.text.as_deref().unwrap_or("");
            let source = doc
                .metadata
                .as_ref()
                .and_then(|m| m.get_string("source").or_else(|| m.get_string("url")))
                .map(|s| s.to_string());

            citations.push(Citation {
                index: citation_idx,
                doc_id: doc.id.to_string(),
                score: doc.score,
                source,
                metadata: doc.metadata.clone().unwrap_or_default(),
            });

            let block = Self::format_block(doc, citation_idx, doc_text, config);
            let block_len = block.len();

            if current_chars + block_len <= config.max_chars {
                assembled_blocks.push(block);
                current_chars += block_len + 2; // account for newline separator
            } else {
                match config.truncation_strategy {
                    TruncationStrategy::DropOversized => {
                        was_truncated = true;
                        citations.pop();
                        break;
                    }
                    TruncationStrategy::TruncateLast => {
                        let remaining = config.max_chars.saturating_sub(current_chars);
                        let header_only = Self::format_block_header(doc, citation_idx, config);
                        if remaining > header_only.len() + 15 {
                            let available_text_chars = remaining - header_only.len() - 10;
                            let truncated_text = if available_text_chars < doc_text.len() {
                                format!("{}... [truncated]", &doc_text[..available_text_chars])
                            } else {
                                doc_text.to_string()
                            };
                            let truncated_block =
                                Self::format_custom_block(&header_only, &truncated_text, config);
                            assembled_blocks.push(truncated_block);
                            was_truncated = true;
                        } else {
                            was_truncated = true;
                            citations.pop();
                        }
                        break;
                    }
                }
            }
        }

        let mut output = String::new();
        if let Some(h) = header_text {
            output.push_str(&h);
        }

        match config.format {
            ContextFormat::Xml => {
                output.push_str("<context>\n");
                for block in &assembled_blocks {
                    output.push_str(block);
                    output.push('\n');
                }
                output.push_str("</context>");
            }
            ContextFormat::Markdown | ContextFormat::Numbered | ContextFormat::Compact => {
                output.push_str(&assembled_blocks.join("\n\n"));
            }
        }

        let doc_count = assembled_blocks.len();
        let total_chars = output.len();

        AssembledContext {
            text: output,
            citations,
            doc_count,
            total_chars,
            was_truncated,
        }
    }

    /// Formats a full document block according to layout choice.
    fn format_block(
        doc: &ScoredDocument,
        index: usize,
        text: &str,
        config: &ContextConfig,
    ) -> String {
        match config.format {
            ContextFormat::Markdown => {
                let score_str = if config.include_scores {
                    format!(" (Score: {:.3})", doc.score)
                } else {
                    String::new()
                };
                let meta_str = Self::format_meta_summary(doc, config);
                format!("### [{}] {}{}{}\n{}", index, doc.id, score_str, meta_str, text)
            }
            ContextFormat::Xml => {
                let score_attr = if config.include_scores {
                    format!(" score=\"{:.3}\"", doc.score)
                } else {
                    String::new()
                };
                format!(
                    "  <document id=\"{}\" index=\"{}\"{}>\n    {}\n  </document>",
                    doc.id, index, score_attr, text
                )
            }
            ContextFormat::Numbered => {
                let score_str = if config.include_scores {
                    format!(" [{:.3}]", doc.score)
                } else {
                    String::new()
                };
                format!("[{}] ({}){}: {}", index, doc.id, score_str, text)
            }
            ContextFormat::Compact => {
                format!("[{}] {}", index, text.replace('\n', " "))
            }
        }
    }

    /// Formats only the block header for truncation budgeting.
    fn format_block_header(doc: &ScoredDocument, index: usize, config: &ContextConfig) -> String {
        match config.format {
            ContextFormat::Markdown => {
                let score_str = if config.include_scores {
                    format!(" (Score: {:.3})", doc.score)
                } else {
                    String::new()
                };
                format!("### [{}] {}{}\n", index, doc.id, score_str)
            }
            ContextFormat::Xml => {
                format!("  <document id=\"{}\" index=\"{}\">\n    ", doc.id, index)
            }
            ContextFormat::Numbered => {
                format!("[{}] ({}): ", index, doc.id)
            }
            ContextFormat::Compact => {
                format!("[{}] ", index)
            }
        }
    }

    /// Combines pre-computed header with custom/truncated text.
    fn format_custom_block(header: &str, text: &str, config: &ContextConfig) -> String {
        match config.format {
            ContextFormat::Xml => format!("{}{}\n  </document>", header, text),
            _ => format!("{}{}", header, text),
        }
    }

    /// Formats a concise metadata summary line if enabled.
    fn format_meta_summary(doc: &ScoredDocument, config: &ContextConfig) -> String {
        if !config.include_metadata {
            return String::new();
        }
        if let Some(ref meta) = doc.metadata {
            if meta.is_empty() {
                return String::new();
            }
            let mut fields = Vec::new();
            if let Some(src) = meta.get_string("source") {
                fields.push(format!("source: {}", src));
            }
            if let Some(cat) = meta.get_string("category") {
                fields.push(format!("category: {}", cat));
            }
            if !fields.is_empty() {
                return format!(" [{}]", fields.join(", "));
            }
        }
        String::new()
    }
}

/// Prompt template assembler wrapping system prompt, context passages, and user query.
#[derive(Debug, Clone)]
pub struct RagPromptTemplate {
    pub system_instruction: String,
    pub user_query_prefix: String,
}

impl Default for RagPromptTemplate {
    fn default() -> Self {
        Self {
            system_instruction:
                "Answer the question accurately based solely on the provided context passages."
                    .to_string(),
            user_query_prefix: "User Query:".to_string(),
        }
    }
}

impl RagPromptTemplate {
    /// Creates a prompt template with custom system instructions.
    pub fn new(system_instruction: impl Into<String>) -> Self {
        Self {
            system_instruction: system_instruction.into(),
            user_query_prefix: "User Query:".to_string(),
        }
    }

    /// Formats the complete final prompt.
    pub fn format_prompt(&self, query: &str, context: &AssembledContext) -> String {
        format!(
            "{}\n\n{}\n\n{} {}",
            self.system_instruction, context.text, self.user_query_prefix, query
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rag::types::DocumentId;

    #[test]
    fn test_context_assembler_markdown_and_budget() {
        let docs = vec![
            ScoredDocument {
                id: DocumentId::new("doc1"),
                score: 0.95,
                dense_score: Some(0.95),
                sparse_score: None,
                text: Some("First document with essential info about LSM trees.".into()),
                metadata: None,
                explanation: None,
            },
            ScoredDocument {
                id: DocumentId::new("doc2"),
                score: 0.85,
                dense_score: Some(0.85),
                sparse_score: None,
                text: Some("Second document with details on write-ahead logs.".into()),
                metadata: None,
                explanation: None,
            },
        ];

        let config = ContextConfig::new()
            .format(ContextFormat::Markdown)
            .include_scores(true)
            .max_chars(250);

        let assembled = ContextAssembler::assemble(&docs, &config);
        assert_eq!(assembled.doc_count, 2);
        assert!(assembled.text.contains("### [1] doc1 (Score: 0.950)"));
        assert!(assembled.text.contains("First document with essential info"));
        assert_eq!(assembled.citations.len(), 2);
        assert_eq!(assembled.citations[0].doc_id, "doc1");
        assert_eq!(assembled.citations[1].doc_id, "doc2");
    }

    #[test]
    fn test_context_assembler_xml_format() {
        let docs = vec![ScoredDocument {
            id: DocumentId::new("doc_xml"),
            score: 0.88,
            dense_score: None,
            sparse_score: Some(0.88),
            text: Some("XML formatted context passage.".into()),
            metadata: None,
            explanation: None,
        }];

        let config = ContextConfig::new()
            .format(ContextFormat::Xml)
            .without_header();

        let assembled = ContextAssembler::assemble(&docs, &config);
        assert!(assembled.text.starts_with("<context>"));
        assert!(assembled.text.ends_with("</context>"));
        assert!(assembled.text.contains("<document id=\"doc_xml\" index=\"1\""));
    }

    #[test]
    fn test_context_assembler_truncation() {
        let docs = vec![
            ScoredDocument {
                id: DocumentId::new("d1"),
                score: 1.0,
                dense_score: None,
                sparse_score: None,
                text: Some("Short text.".into()),
                metadata: None,
                explanation: None,
            },
            ScoredDocument {
                id: DocumentId::new("d2"),
                score: 0.8,
                dense_score: None,
                sparse_score: None,
                text: Some("Very long text that should definitely be truncated because the budget is tiny.".into()),
                metadata: None,
                explanation: None,
            },
        ];

        let config = ContextConfig::new()
            .without_header()
            .max_chars(95)
            .truncation_strategy(TruncationStrategy::TruncateLast);

        let assembled = ContextAssembler::assemble(&docs, &config);
        assert!(assembled.was_truncated);
        assert!(assembled.text.contains("[truncated]"));
    }
}
