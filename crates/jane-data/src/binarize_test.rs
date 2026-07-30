use std::io::{Read, Write};

use tempfile::{NamedTempFile, tempdir};

use super::*;
use crate::tokenizer::{EOT_TOKEN, JaneTokenizer};

/// Build a small tokenizer trained on some simple text.
fn make_tokenizer() -> JaneTokenizer {
    let content = "hello world foo bar baz\n".repeat(200)
        + &"the quick brown fox jumps over the lazy dog\n".repeat(200);
    let mut f = NamedTempFile::new().unwrap();
    f.write_all(content.as_bytes()).unwrap();
    f.flush().unwrap();
    JaneTokenizer::train_from_files(&[f.path()], 300).unwrap()
}

fn write_text_file(content: &str) -> NamedTempFile {
    let mut f = NamedTempFile::new().unwrap();
    f.write_all(content.as_bytes()).unwrap();
    f.flush().unwrap();
    f
}

/// Read a .bin file into a vector of u16 token ids.
fn read_bin(path: &std::path::Path) -> Vec<u16> {
    let mut data = Vec::new();
    std::fs::File::open(path)
        .unwrap()
        .read_to_end(&mut data)
        .unwrap();
    assert_eq!(data.len() % 2, 0, "bin file has odd byte count");
    data.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

// -------------------------------------------------------------------------
// Two documents, separated by EOT marker
// -------------------------------------------------------------------------

#[test]
fn two_docs_produce_two_eot_tokens() {
    let tok = make_tokenizer();
    let eot = tok.eot_id() as u16;

    let content = format!("hello world{EOT_TOKEN}foo bar");
    let in_file = write_text_file(&content);
    let dir = tempdir().unwrap();
    let out = dir.path().join("out.bin");

    let stats = binarize_text_file(in_file.path(), &tok, &out, Some(EOT_TOKEN)).unwrap();

    assert_eq!(stats.docs, 2, "expected 2 docs, got {}", stats.docs);

    let ids = read_bin(&out);
    let eot_count = ids.iter().filter(|&&x| x == eot).count();
    assert_eq!(eot_count, 2, "expected 2 EOT tokens, got {eot_count}");
}

// -------------------------------------------------------------------------
// tokens == bin_length / 2 and sidecar agrees
// -------------------------------------------------------------------------

#[test]
fn token_count_matches_bin_length_and_sidecar() {
    let tok = make_tokenizer();
    let in_file = write_text_file("hello world");
    let dir = tempdir().unwrap();
    let out = dir.path().join("out.bin");

    let stats = binarize_text_file(in_file.path(), &tok, &out, None).unwrap();

    let bin_len = std::fs::metadata(&out).unwrap().len();
    assert_eq!(
        stats.tokens,
        bin_len / 2,
        "stats.tokens doesn't match bin length"
    );

    let meta = crate::meta::TokenMeta::load_for(&out).unwrap();
    meta.check_bin_len(&out, bin_len).unwrap();
}

// -------------------------------------------------------------------------
// Little-endian byte order
// -------------------------------------------------------------------------

#[test]
fn written_bytes_are_little_endian() {
    let tok = make_tokenizer();
    let text = "hello";
    let in_file = write_text_file(text);
    let dir = tempdir().unwrap();
    let out = dir.path().join("out.bin");

    binarize_text_file(in_file.path(), &tok, &out, None).unwrap();

    let expected_ids = tok.encode(text).unwrap();
    let eot = tok.eot_id();
    let mut expected_bytes: Vec<u8> = expected_ids
        .iter()
        .flat_map(|&id| (id as u16).to_le_bytes())
        .collect();
    expected_bytes.extend_from_slice(&(eot as u16).to_le_bytes());

    let mut actual_bytes = Vec::new();
    std::fs::File::open(&out)
        .unwrap()
        .read_to_end(&mut actual_bytes)
        .unwrap();

    assert_eq!(actual_bytes, expected_bytes, "byte order mismatch");
}

// -------------------------------------------------------------------------
// Empty input file
// -------------------------------------------------------------------------

#[test]
fn empty_input_produces_valid_output() {
    let tok = make_tokenizer();
    let in_file = write_text_file("");
    let dir = tempdir().unwrap();
    let out = dir.path().join("out.bin");

    let stats = binarize_text_file(in_file.path(), &tok, &out, None).unwrap();

    // An empty file has no content, so no tokens and no EOT.
    assert_eq!(stats.docs, 0);
    assert_eq!(stats.tokens, 0);
    let bin_len = std::fs::metadata(&out).unwrap().len();
    assert_eq!(bin_len, 0, "empty input should produce 0-byte bin");

    // Sidecar must still be valid.
    let meta = crate::meta::TokenMeta::load_for(&out).unwrap();
    meta.check_bin_len(&out, bin_len).unwrap();
}

// -------------------------------------------------------------------------
// doc_sep: None emits exactly one EOT
// -------------------------------------------------------------------------

#[test]
fn none_doc_sep_emits_exactly_one_eot() {
    let tok = make_tokenizer();
    let eot = tok.eot_id() as u16;
    let in_file = write_text_file("hello world foo bar");
    let dir = tempdir().unwrap();
    let out = dir.path().join("out.bin");

    let stats = binarize_text_file(in_file.path(), &tok, &out, None).unwrap();

    assert_eq!(stats.docs, 1);
    let ids = read_bin(&out);
    let eot_count = ids.iter().filter(|&&x| x == eot).count();
    assert_eq!(eot_count, 1, "expected exactly 1 EOT, got {eot_count}");
}

// -------------------------------------------------------------------------
// Consecutive separators do not emit empty documents
// -------------------------------------------------------------------------

#[test]
fn consecutive_separators_skip_empty_docs() {
    let tok = make_tokenizer();
    // Three separators in a row between two real docs.
    let content = format!("doc one{EOT_TOKEN}{EOT_TOKEN}{EOT_TOKEN}doc two");
    let in_file = write_text_file(&content);
    let dir = tempdir().unwrap();
    let out = dir.path().join("out.bin");

    let stats = binarize_text_file(in_file.path(), &tok, &out, Some(EOT_TOKEN)).unwrap();

    assert_eq!(
        stats.docs, 2,
        "consecutive separators must not produce empty docs; got {}",
        stats.docs
    );
}

// -------------------------------------------------------------------------
// Streaming correctness: documents straddling chunk boundaries
// -------------------------------------------------------------------------

#[test]
fn streaming_matches_per_doc_tokenization() {
    let tok = make_tokenizer();
    let eot = tok.eot_id() as u16;

    // Build a text with many documents so separators straddle read
    // boundaries (each line read by BufReader). 500 docs of ~30 bytes each.
    let separator = EOT_TOKEN;
    let docs: Vec<String> = (0..500u32)
        .map(|i| format!("hello world doc {i} foo bar"))
        .collect();
    let full_text = docs.join(separator);

    let in_file = write_text_file(&full_text);
    let dir = tempdir().unwrap();
    let out = dir.path().join("stream.bin");

    let stats = binarize_text_file(in_file.path(), &tok, &out, Some(separator)).unwrap();

    assert_eq!(stats.docs, 500, "expected 500 docs, got {}", stats.docs);

    let ids = read_bin(&out);
    let eot_count = ids.iter().filter(|&&x| x == eot).count();
    assert_eq!(eot_count, 500, "expected 500 EOT tokens, got {eot_count}");

    // Cross-check: independently tokenize each doc and verify content
    // token counts match.
    let expected_content_tokens: usize =
        docs.iter().map(|d| tok.encode(d).unwrap().len()).sum();
    let actual_content_tokens = ids.iter().filter(|&&x| x != eot).count();
    assert_eq!(
        actual_content_tokens, expected_content_tokens,
        "streaming content token count differs from per-doc tokenization"
    );
}

// -------------------------------------------------------------------------
// Sidecar records vocab_size
// -------------------------------------------------------------------------

#[test]
fn sidecar_records_vocab_size() {
    let tok = make_tokenizer();
    let in_file = write_text_file("hello world");
    let dir = tempdir().unwrap();
    let out = dir.path().join("out.bin");

    binarize_text_file(in_file.path(), &tok, &out, None).unwrap();

    let meta = crate::meta::TokenMeta::load_for(&out).unwrap();
    assert_eq!(meta.vocab_size, tok.vocab_size());
}
