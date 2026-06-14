use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use sochdb_query::context_query::{EmbeddingProvider, SimpleVectorIndex, VectorIndex};
use sochdb_query::soch_ql::SochValue;

#[derive(Debug, Clone, Deserialize)]
struct JsonlRow {
    conversation_id: Option<String>,
    session_id: Option<String>,
    turn: Option<u64>,
    role: Option<String>,
    text: Option<String>,
    timestamp: Option<String>,
    memory_type: Option<String>,
    importance: Option<f64>,

    question_id: Option<String>,
    question: Option<String>,
    answer: Option<String>,
    evidence_turns: Option<Vec<u64>>,

    #[serde(rename = "type")]
    qtype: Option<String>,
}

#[derive(Debug, Serialize)]
struct RetrievalResult {
    system: String,
    question_id: String,
    question: String,
    gold_answer: String,
    retrieved_turns: Vec<u64>,
    evidence_turns: Vec<u64>,
    retrieved_count: usize,
    approx_context_tokens: usize,
    latency_ms: f64,
    debug_context: String,

    #[serde(rename = "type")]
    qtype: Option<String>,
}

/// Deterministic local embedding provider.
///
/// This is NOT a semantic model. It is a reproducible hash-bag embedding
/// so we can test the real SochDB vector index path without an external model.
struct HashEmbeddingProvider {
    dim: usize,
}

impl HashEmbeddingProvider {
    fn new(dim: usize) -> Self {
        Self { dim }
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let mut vector = vec![0.0_f32; self.dim];

        for token in normalize(text).split_whitespace() {
            let idx = stable_hash(token) % self.dim;
            vector[idx] += 1.0;
        }

        // L2 normalize.
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm > 0.0 {
            for x in &mut vector {
                *x /= norm;
            }
        }

        vector
    }
}

impl EmbeddingProvider for HashEmbeddingProvider {
    fn embed_text(&self, text: &str) -> Result<Vec<f32>, String> {
        Ok(self.embed(text))
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        "deterministic-hash-embedding"
    }
}

fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());

    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() || ch.is_whitespace() {
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push(' ');
        }
    }

    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn stable_hash(s: &str) -> usize {
    // FNV-1a 64-bit.
    let mut hash: u64 = 0xcbf29ce484222325;

    for b in s.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }

    hash as usize
}

fn approx_tokens(s: &str) -> usize {
    s.split_whitespace().count()
}

fn parse_args() -> Result<(PathBuf, usize, PathBuf), String> {
    let args: Vec<String> = env::args().collect();

    let mut data: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut k: usize = 3;

    let mut i = 1;

    while i < args.len() {
        match args[i].as_str() {
            "--data" => {
                i += 1;
                data = args.get(i).map(PathBuf::from);
            }
            "--k" => {
                i += 1;
                k = args
                    .get(i)
                    .ok_or("--k requires a value")?
                    .parse()
                    .map_err(|e| format!("invalid --k value: {e}"))?;
            }
            "--out" => {
                i += 1;
                out = args.get(i).map(PathBuf::from);
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run -p sochdb-query --example agent_memory_qa_context_select -- \\
  --data <small_memory_qa.jsonl> \\
  --k <top_k> \\
  --out <retrieval.jsonl>"
                );
                std::process::exit(0);
            }
            other => {
                return Err(format!("unknown argument: {other}"));
            }
        }

        i += 1;
    }

    let data = data.ok_or("missing --data")?;
    let out = out.ok_or("missing --out")?;

    Ok((data, k, out))
}

fn read_jsonl(path: &PathBuf) -> Result<Vec<JsonlRow>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;

    let mut rows = Vec::new();

    for (line_no, line) in raw.lines().enumerate() {
        let line = line.trim();

        if line.is_empty() {
            continue;
        }

        let row: JsonlRow = serde_json::from_str(line)
            .map_err(|e| format!("invalid JSONL at {}:{}: {e}", path.display(), line_no + 1))?;

        rows.push(row);
    }

    Ok(rows)
}

fn main() -> Result<(), String> {
    let (data_path, k, out_path) = parse_args()?;

    let rows = read_jsonl(&data_path)?;

    let memories: Vec<JsonlRow> = rows
        .iter()
        .filter(|row| row.question.is_none())
        .cloned()
        .collect();

    let questions: Vec<JsonlRow> = rows
        .iter()
        .filter(|row| row.question.is_some())
        .cloned()
        .collect();

    if memories.is_empty() {
        return Err("no memory rows found in JSONL".to_string());
    }

    if questions.is_empty() {
        return Err("no question rows found in JSONL".to_string());
    }

    let dim = 512;
    let collection = "agent_memory";

    let embedder = HashEmbeddingProvider::new(dim);

    // This uses the real SimpleVectorIndex implementation from context_query.rs.
    let index = SimpleVectorIndex::new();
    index.create_collection(collection, dim);

    // Insert memory rows into the vector index.
    for memory in &memories {
        let turn = memory.turn.unwrap_or(0);
        let text = memory.text.clone().unwrap_or_default();

        let embedding = embedder.embed_text(&text)?;

        let mut metadata = HashMap::new();

        metadata.insert("turn".to_string(), SochValue::Int(turn as i64));

        if let Some(conversation_id) = &memory.conversation_id {
            metadata.insert(
                "conversation_id".to_string(),
                SochValue::Text(conversation_id.clone()),
            );
        }

        if let Some(session_id) = &memory.session_id {
            metadata.insert("session_id".to_string(), SochValue::Text(session_id.clone()));
        }

        if let Some(role) = &memory.role {
            metadata.insert("role".to_string(), SochValue::Text(role.clone()));
        }

        if let Some(timestamp) = &memory.timestamp {
            metadata.insert("timestamp".to_string(), SochValue::Text(timestamp.clone()));
        }

        if let Some(memory_type) = &memory.memory_type {
            metadata.insert(
                "memory_type".to_string(),
                SochValue::Text(memory_type.clone()),
            );
        }

        if let Some(importance) = memory.importance {
            metadata.insert("importance".to_string(), SochValue::Float(importance));
        }

        index.insert(
            collection,
            format!("turn_{turn}"),
            embedding,
            text,
            metadata,
        )?;
    }

    let mut output_rows = Vec::new();

    for question_row in questions {
        let question_id = question_row
            .question_id
            .clone()
            .ok_or("question row missing question_id")?;

        let question = question_row
            .question
            .clone()
            .ok_or("question row missing question")?;

        let gold_answer = question_row.answer.clone().unwrap_or_default();
        let evidence_turns = question_row.evidence_turns.clone().unwrap_or_default();

        let start = Instant::now();

        let question_embedding = embedder.embed_text(&question)?;

        let hits = index.search_by_embedding(
            collection,
            &question_embedding,
            k,
            None,
        )?;

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        let mut retrieved_turns = Vec::new();
        let mut context_parts = Vec::new();

        for hit in hits {
            let turn = hit.metadata.get("turn").and_then(|value| match value {
                SochValue::Int(n) => Some(*n as u64),
                SochValue::Text(s) => s.parse::<u64>().ok(),
                _ => None,
            });

            if let Some(turn) = turn {
                retrieved_turns.push(turn);
            }

            context_parts.push(hit.content);
        }

        let debug_context = context_parts.join("\n");

        output_rows.push(RetrievalResult {
            system: "context_select_search".to_string(),
            question_id,
            question,
            gold_answer,
            retrieved_turns,
            evidence_turns,
            retrieved_count: context_parts.len(),
            approx_context_tokens: approx_tokens(&debug_context),
            latency_ms,
            debug_context,
            qtype: question_row.qtype.clone(),
        });
    }

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    }

    let mut out = String::new();

    for row in output_rows {
        out.push_str(
            &serde_json::to_string(&row)
                .map_err(|e| format!("failed to serialize output row: {e}"))?,
        );
        out.push('\n');
    }

    fs::write(&out_path, out)
        .map_err(|e| format!("failed to write {}: {e}", out_path.display()))?;

    eprintln!("wrote {}", out_path.display());

    Ok(())
}