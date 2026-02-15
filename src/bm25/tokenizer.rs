//! Whitespace tokenizer with stop word removal.
//!
//! Tokenizes text by lowercasing, splitting on non-alphanumeric characters,
//! and removing common English and Italian stop words. Single-character tokens
//! are also discarded. Uses a zero-per-token allocation design via byte spans.

use std::collections::HashSet;
use std::sync::LazyLock;

static STOP_WORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is",
        "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there",
        "these", "they", "this", "to", "was", "will", "with",
        // Italian common stop words
        "il", "lo", "la", "le", "li", "gli", "un", "uno", "una", "di", "da", "del", "dello",
        "della", "dei", "degli", "delle", "che", "per", "non", "si", "con", "su", "ha", "sono",
        "come", "ma", "anche", "se", "io", "ci",
    ]
    .into_iter()
    .collect()
});

/// Tokenized text: owns the lowercased buffer, provides &str slices via byte spans.
/// Only 1 heap allocation (the lowercased String) instead of N per-token Strings.
pub struct Tokens {
    buffer: String,
    spans: Vec<(u32, u32)>, // (start, end) byte offsets into buffer
}

impl Tokens {
    /// Returns an iterator over the token `&str` slices.
    pub fn iter(&self) -> impl Iterator<Item = &str> + '_ {
        self.spans
            .iter()
            .map(|&(s, e)| &self.buffer[s as usize..e as usize])
    }

    /// Returns the number of tokens.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Returns `true` if there are no tokens.
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }
}

/// Tokenize text: lowercase, split on non-alphanumeric, remove stop words.
/// Returns a Tokens struct that owns the lowercased buffer. Zero per-token allocation.
pub fn tokenize(text: &str) -> Tokens {
    let buffer = text.to_lowercase();
    let mut spans = Vec::new();
    let mut start: Option<usize> = None;

    for (i, c) in buffer.char_indices() {
        if c.is_alphanumeric() {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start {
            let token = &buffer[s..i];
            if token.len() > 1 && !STOP_WORDS.contains(token) {
                spans.push((s as u32, i as u32));
            }
            start = None;
        }
    }
    // Handle last token (no trailing separator)
    if let Some(s) = start {
        let token = &buffer[s..];
        if token.len() > 1 && !STOP_WORDS.contains(token) {
            spans.push((s as u32, buffer.len() as u32));
        }
    }

    Tokens { buffer, spans }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("The quick brown fox jumps over the lazy dog");
        let words: Vec<&str> = tokens.iter().collect();
        assert!(!words.contains(&"the"));
        assert!(words.contains(&"quick"));
        assert!(words.contains(&"brown"));
        assert!(words.contains(&"fox"));
    }
}
