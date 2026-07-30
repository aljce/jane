use std::io::Write;

use tempfile::NamedTempFile;

use super::*;

/// Write a temporary file with the given content and return it (kept alive).
fn temp_file_with(content: &str) -> NamedTempFile {
    let mut f = NamedTempFile::new().unwrap();
    f.write_all(content.as_bytes()).unwrap();
    f.flush().unwrap();
    f
}

fn train_small_tokenizer() -> JaneTokenizer {
    // Repetitive text gives the trainer enough bigrams to actually merge.
    let content = "hello world hello world foo bar baz qux\n".repeat(200)
        + &"the quick brown fox jumps over the lazy dog\n".repeat(200);
    let f = temp_file_with(&content);
    JaneTokenizer::train_from_files(&[f.path()], 300).unwrap()
}

// --- train_from_files tests ---

#[test]
fn train_vocab_size_bounds() {
    let tok = train_small_tokenizer();
    let vs = tok.vocab_size();
    // The tokenizer may stop early if it runs out of merges, but must be
    // at least 256 (byte alphabet) and at most what we requested.
    assert!(vs >= 256, "vocab_size {vs} < 256");
    assert!(vs <= 300, "vocab_size {vs} > requested 300");
}

#[test]
fn train_all_ids_in_range() {
    let tok = train_small_tokenizer();
    let ids = tok.encode("hello world foo bar").unwrap();
    let vs = tok.vocab_size() as u32;
    for id in &ids {
        assert!(*id < vs, "id {id} >= vocab_size {vs}");
    }
}

#[test]
fn train_rejects_vocab_below_256() {
    let f = temp_file_with("abc");
    let err = JaneTokenizer::train_from_files(&[f.path()], 255).unwrap_err();
    assert!(
        matches!(err, DataError::Tokenizer(_)),
        "expected Tokenizer error, got {err:?}"
    );
}

// --- save / load ---

#[test]
fn save_load_round_trip_encodes_identically() {
    let tok = train_small_tokenizer();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.json");

    tok.save(&path).unwrap();
    let reloaded = JaneTokenizer::load(&path).unwrap();

    let sample = "hello world";
    assert_eq!(
        tok.encode(sample).unwrap(),
        reloaded.encode(sample).unwrap()
    );
}

// --- encode / decode round-trip ---

fn assert_round_trip(tok: &JaneTokenizer, s: &str) {
    let ids = tok.encode(s).unwrap();
    let decoded = tok.decode(&ids).unwrap();
    assert_eq!(
        decoded, s,
        "round-trip failed for {:?}: ids={ids:?}, decoded={decoded:?}",
        s
    );
}

#[test]
fn round_trip_plain_ascii() {
    let tok = train_small_tokenizer();
    assert_round_trip(&tok, "hello world");
}

#[test]
fn round_trip_empty_string() {
    let tok = train_small_tokenizer();
    let ids = tok.encode("").unwrap();
    assert!(
        ids.is_empty(),
        "empty string should produce no ids, got {ids:?}"
    );
    let decoded = tok.decode(&[]).unwrap();
    assert_eq!(decoded, "");
}

#[test]
fn round_trip_whitespace() {
    let tok = train_small_tokenizer();
    assert_round_trip(&tok, "  leading");
    assert_round_trip(&tok, "trailing  ");
    assert_round_trip(&tok, "  both  ");
    assert_round_trip(&tok, "  multiple   spaces  ");
}

#[test]
fn round_trip_non_ascii() {
    let tok = train_small_tokenizer();
    assert_round_trip(&tok, "héllo wörld");
    assert_round_trip(&tok, "日本語のテキスト");
    assert_round_trip(&tok, "🙂🙃");
}

#[test]
fn round_trip_eot_token_literal() {
    let tok = train_small_tokenizer();
    assert_round_trip(&tok, EOT_TOKEN);
}

#[test]
fn round_trip_newline_tab() {
    let tok = train_small_tokenizer();
    assert_round_trip(&tok, "line1\nline2\ttab");
}

// --- eot_id ---

#[test]
fn eot_id_in_range() {
    let tok = train_small_tokenizer();
    let eot = tok.eot_id();
    let vs = tok.vocab_size() as u32;
    assert!(eot < vs, "eot_id {eot} >= vocab_size {vs}");
}

#[test]
fn eot_id_decodes_to_empty() {
    let tok = train_small_tokenizer();
    let eot = tok.eot_id();
    // Special tokens are skipped when decoding.
    let decoded = tok.decode(&[eot]).unwrap();
    assert_eq!(
        decoded, "",
        "decoding [eot_id] should yield empty string, got {decoded:?}"
    );
}

// --- sha256_of_file ---

#[test]
fn sha256_empty_file_known_hash() {
    let f = temp_file_with("");
    let hash = sha256_of_file(f.path()).unwrap();
    assert_eq!(
        hash,
        "e3b0c44298fc1c149afbf4c8996fb924\
         27ae41e4649b934ca495991b7852b855"
    );
}

#[test]
fn sha256_identical_files_equal_hash() {
    let content = "the quick brown fox";
    let f1 = temp_file_with(content);
    let f2 = temp_file_with(content);
    assert_eq!(
        sha256_of_file(f1.path()).unwrap(),
        sha256_of_file(f2.path()).unwrap()
    );
}

#[test]
fn sha256_one_byte_change_differs() {
    let f1 = temp_file_with("hello");
    let f2 = temp_file_with("heLlo");
    assert_ne!(
        sha256_of_file(f1.path()).unwrap(),
        sha256_of_file(f2.path()).unwrap()
    );
}
