//! # Document Chunking Algorithms
//!
//! Provides chunking strategies to decompose long documents into manageable
//! passages/chunks with overlap, span tracking, and metadata preservation.

use crate::rag::types::{ChunkingConfig, ChunkingStrategy, Document, DocumentChunk};

/// Chunks raw text according to the provided `ChunkingConfig`.
///
/// Returns a list of `(start_char_offset, end_char_offset, chunk_text)`.
pub fn chunk_text(text: &str, config: &ChunkingConfig) -> Vec<(usize, usize, String)> {
    if text.trim().is_empty() {
        return Vec::new();
    }

    match config.strategy {
        ChunkingStrategy::FixedChars { size, overlap } => {
            chunk_fixed_chars(text, size, overlap, config.min_chunk_size)
        }
        ChunkingStrategy::FixedTokens { size, overlap } => {
            chunk_fixed_tokens(text, size, overlap, config.min_chunk_size)
        }
        ChunkingStrategy::Paragraph => chunk_paragraphs(text, config.min_chunk_size),
        ChunkingStrategy::Sentence => chunk_sentences(text, config.min_chunk_size),
    }
}

/// Chunks text using fixed character windows with overlap.
fn chunk_fixed_chars(
    text: &str,
    size: usize,
    overlap: usize,
    min_chunk_size: usize,
) -> Vec<(usize, usize, String)> {
    if size == 0 {
        return Vec::new();
    }

    let char_indices: Vec<(usize, char)> = text.char_indices().collect();
    let total_chars = char_indices.len();
    if total_chars == 0 {
        return Vec::new();
    }

    let effective_overlap = overlap.min(size.saturating_sub(1));
    let step = size.saturating_sub(effective_overlap).max(1);

    let mut chunks = Vec::new();
    let mut c_start = 0;

    while c_start < total_chars {
        let c_end = (c_start + size).min(total_chars);
        let byte_start = char_indices[c_start].0;
        let byte_end = if c_end < total_chars {
            char_indices[c_end].0
        } else {
            text.len()
        };

        let chunk_slice = &text[byte_start..byte_end];
        let trimmed = chunk_slice.trim();

        if trimmed.len() >= min_chunk_size || (chunks.is_empty() && c_end >= total_chars) {
            chunks.push((byte_start, byte_end, chunk_slice.to_string()));
        }

        if c_end >= total_chars {
            break;
        }
        c_start += step;
    }

    chunks
}

/// Chunks text using fixed token (word) windows with overlap.
fn chunk_fixed_tokens(
    text: &str,
    size: usize,
    overlap: usize,
    min_chunk_size: usize,
) -> Vec<(usize, usize, String)> {
    if size == 0 {
        return Vec::new();
    }

    let mut token_spans: Vec<(usize, usize)> = Vec::new();
    let mut start_idx = None;

    for (byte_idx, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if let Some(s) = start_idx.take() {
                token_spans.push((s, byte_idx));
            }
        } else if start_idx.is_none() {
            start_idx = Some(byte_idx);
        }
    }
    if let Some(s) = start_idx {
        token_spans.push((s, text.len()));
    }

    if token_spans.is_empty() {
        return Vec::new();
    }

    let effective_overlap = overlap.min(size.saturating_sub(1));
    let step = size.saturating_sub(effective_overlap).max(1);

    let mut chunks = Vec::new();
    let mut t_start = 0;

    while t_start < token_spans.len() {
        let t_end = (t_start + size).min(token_spans.len());
        let byte_start = token_spans[t_start].0;
        let byte_end = token_spans[t_end - 1].1;

        let chunk_slice = &text[byte_start..byte_end];
        let trimmed = chunk_slice.trim();

        if trimmed.len() >= min_chunk_size || (chunks.is_empty() && t_end >= token_spans.len()) {
            chunks.push((byte_start, byte_end, chunk_slice.to_string()));
        }

        if t_end >= token_spans.len() {
            break;
        }
        t_start += step;
    }

    chunks
}

/// Chunks text into paragraphs separated by two or more consecutive newlines.
fn chunk_paragraphs(text: &str, min_chunk_size: usize) -> Vec<(usize, usize, String)> {
    let mut chunks = Vec::new();
    let mut para_start = None;
    let mut consecutive_newlines = 0;

    let chars: Vec<(usize, char)> = text.char_indices().collect();

    for &(byte_idx, ch) in &chars {
        if ch == '\n' {
            consecutive_newlines += 1;
            if consecutive_newlines >= 2 {
                if let Some(s) = para_start.take() {
                    let slice = &text[s..byte_idx];
                    let trimmed = slice.trim();
                    if trimmed.len() >= min_chunk_size {
                        chunks.push((s, byte_idx, trimmed.to_string()));
                    }
                }
            }
        } else {
            if !ch.is_whitespace() && para_start.is_none() {
                para_start = Some(byte_idx);
            }
            consecutive_newlines = 0;
        }
    }

    if let Some(s) = para_start {
        let slice = &text[s..text.len()];
        let trimmed = slice.trim();
        if trimmed.len() >= min_chunk_size || chunks.is_empty() {
            chunks.push((s, text.len(), trimmed.to_string()));
        }
    }

    chunks
}

/// Chunks text into sentences based on punctuation delimiters (`. `, `! `, `? `, `\n`).
fn chunk_sentences(text: &str, min_chunk_size: usize) -> Vec<(usize, usize, String)> {
    let mut chunks = Vec::new();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let len = chars.len();
    if len == 0 {
        return Vec::new();
    }

    let mut start = 0;
    let mut i = 0;

    while i < len {
        let (_byte_idx, ch) = chars[i];
        let is_sentence_end = match ch {
            '.' | '!' | '?' => {
                // Check if next character is whitespace, newline, quote, or EOF
                if i + 1 < len {
                    let next_ch = chars[i + 1].1;
                    next_ch.is_whitespace() || next_ch == '"' || next_ch == '\''
                } else {
                    true
                }
            }
            '\n' => true,
            _ => false,
        };

        if is_sentence_end {
            let next_byte = if i + 1 < len {
                chars[i + 1].0
            } else {
                text.len()
            };
            let slice = &text[start..next_byte];
            let trimmed = slice.trim();
            if trimmed.len() >= min_chunk_size {
                chunks.push((start, next_byte, trimmed.to_string()));
                start = next_byte;
            }
        }
        i += 1;
    }

    if start < text.len() {
        let slice = &text[start..text.len()];
        let trimmed = slice.trim();
        if !trimmed.is_empty() {
            if let Some(last) = chunks.last_mut() {
                if trimmed.len() < min_chunk_size {
                    // Merge trailing fragment into last chunk
                    last.1 = text.len();
                    last.2.push(' ');
                    last.2.push_str(trimmed);
                } else {
                    chunks.push((start, text.len(), trimmed.to_string()));
                }
            } else {
                chunks.push((start, text.len(), trimmed.to_string()));
            }
        }
    }

    chunks
}

/// Decomposes a [`Document`] into [`DocumentChunk`] instances according to `config`.
pub fn chunk_document(doc: &Document, config: &ChunkingConfig) -> Vec<DocumentChunk> {
    let raw_chunks = chunk_text(&doc.text, config);
    raw_chunks
        .into_iter()
        .enumerate()
        .map(|(idx, (start_char, end_char, chunk_text))| DocumentChunk {
            chunk_id: format!("{}_chunk_{}", doc.id, idx),
            doc_id: doc.id.clone(),
            chunk_index: idx,
            text: chunk_text,
            embedding: None,
            start_char,
            end_char,
            metadata: doc.metadata.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rag::types::DocumentId;

    #[test]
    fn test_chunk_fixed_chars() {
        let config = ChunkingConfig {
            strategy: ChunkingStrategy::FixedChars {
                size: 20,
                overlap: 5,
            },
            min_chunk_size: 5,
        };
        let text = "FlashStore is an embedded LSM-tree storage engine in Rust.";
        let chunks = chunk_text(text, &config);
        assert!(chunks.len() >= 3);
        assert_eq!(chunks[0].0, 0);
        assert_eq!(&text[chunks[0].0..chunks[0].1], "FlashStore is an emb");
    }

    #[test]
    fn test_chunk_fixed_tokens() {
        let config = ChunkingConfig {
            strategy: ChunkingStrategy::FixedTokens {
                size: 4,
                overlap: 1,
            },
            min_chunk_size: 5,
        };
        let text = "one two three four five six seven eight nine ten";
        let chunks = chunk_text(text, &config);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].2, "one two three four");
        assert_eq!(chunks[1].2, "four five six seven");
        assert_eq!(chunks[2].2, "seven eight nine ten");
    }

    #[test]
    fn test_chunk_paragraphs() {
        let config = ChunkingConfig {
            strategy: ChunkingStrategy::Paragraph,
            min_chunk_size: 10,
        };
        let text = "Paragraph one has sufficient length.\n\nParagraph two is also very long and detailed.\n\nParagraph three.";
        let chunks = chunk_text(text, &config);
        assert_eq!(chunks.len(), 3);
        assert!(chunks[0].2.contains("Paragraph one"));
        assert!(chunks[1].2.contains("Paragraph two"));
        assert!(chunks[2].2.contains("Paragraph three"));
    }

    #[test]
    fn test_chunk_sentences() {
        let config = ChunkingConfig {
            strategy: ChunkingStrategy::Sentence,
            min_chunk_size: 10,
        };
        let text =
            "First sentence here! Second sentence follows. Is this the third sentence? Yes it is.";
        let chunks = chunk_text(text, &config);
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0].2, "First sentence here!");
        assert_eq!(chunks[1].2, "Second sentence follows.");
        assert_eq!(chunks[2].2, "Is this the third sentence?");
        assert_eq!(chunks[3].2, "Yes it is.");
    }

    #[test]
    fn test_chunk_document() {
        let config = ChunkingConfig {
            strategy: ChunkingStrategy::FixedTokens {
                size: 3,
                overlap: 0,
            },
            min_chunk_size: 5,
        };
        let doc = Document::new("doc_100", "apple banana cherry date elderberry fig");
        let chunks = chunk_document(&doc, &config);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].chunk_id, "doc_100_chunk_0");
        assert_eq!(chunks[0].doc_id, DocumentId::new("doc_100"));
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks[0].text, "apple banana cherry");
        assert_eq!(chunks[1].chunk_id, "doc_100_chunk_1");
        assert_eq!(chunks[1].chunk_index, 1);
        assert_eq!(chunks[1].text, "date elderberry fig");
    }
}
