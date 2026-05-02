use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use sha1::{Digest, Sha1};
use std::collections::HashMap;

const AZ: &[u8; 26] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const ENGLISH_FREQ_ORDER: &[u8; 26] = b"ETAOINSHRDLUCMWFGYPBVKJXQZ";
const ENGLISH_FREQS: [f64; 26] = [
    0.0817, 0.0149, 0.0278, 0.0425, 0.1270, 0.0223, 0.0202, 0.0609, 0.0697, 0.0015, 0.0077, 0.0403,
    0.0241, 0.0675, 0.0751, 0.0193, 0.0010, 0.0599, 0.0633, 0.0906, 0.0276, 0.0098, 0.0236, 0.0015,
    0.0197, 0.0007,
];

#[derive(Clone)]
struct Candidate {
    period: usize,
    shifts: Vec<usize>,
    alphabet: [u8; 26],
    initial_alphabet: [u8; 26],
    restart: usize,
    score: f64,
    selection_score: f64,
    init_score: f64,
    plaintext: String,
    guided_proposals: usize,
    random_proposals: usize,
}

#[derive(Clone)]
struct NGramTable {
    probs: HashMap<Vec<u8>, f64>,
    floor: f64,
}

#[derive(Clone)]
struct DenseNGramTable {
    n: usize,
    probs: Vec<f64>,
    floor: f64,
}

#[derive(Clone)]
struct QuagmireCandidate {
    period: usize,
    keyword_len: usize,
    restart: usize,
    hillclimb_step: usize,
    alphabet: [u8; 26],
    cycle_shifts: Vec<usize>,
    score: f64,
    plaintext: String,
    accepted_mutations: usize,
    slips: usize,
    backtracks: usize,
}

#[derive(Clone)]
enum FastValue {
    Int(i64),
    Str(String),
    IntList(Vec<usize>),
}

#[derive(Clone)]
struct FastTransformStep {
    name: String,
    data: HashMap<String, FastValue>,
}

#[derive(Clone)]
struct FastTransformPipeline {
    columns: Option<usize>,
    rows: Option<usize>,
    steps: Vec<FastTransformStep>,
}

#[derive(Clone)]
struct FastTransformCandidate {
    pipeline: FastTransformPipeline,
    grid_columns: Option<usize>,
    grid_rows: Option<usize>,
}

#[derive(Clone)]
struct FastTransformScore {
    valid: bool,
    reason: String,
    token_order_hash: Option<String>,
    position_order_preview: Vec<usize>,
    transformed_preview: Vec<usize>,
    metrics: Option<FastTransformMetrics>,
}

#[derive(Clone)]
struct FastTransformMetrics {
    repeated_bigram_rate: f64,
    repeated_trigram_rate: f64,
    alternation_rate: f64,
    symbol_ioc: f64,
    position_nontriviality: f64,
    position_adjacent_rate: f64,
    position_step_repeat_rate: f64,
    periodic_redundancy: f64,
    inverse_periodic_redundancy: f64,
    best_period: f64,
    inverse_best_period: f64,
    grid_row_step_rate: f64,
    grid_column_step_rate: f64,
    periodic_structure_score: f64,
    matrix_rank_score: f64,
}

#[pyfunction]
fn normalized_ngram_score(text: &str, log_probs: &PyDict, n: usize) -> PyResult<f64> {
    let table = table_from_pydict(log_probs)?;
    Ok(normalized_score(text.as_bytes(), &table, n))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn keyed_vigenere_alphabet_anneal(
    py: Python<'_>,
    symbol_values: Vec<usize>,
    log_probs: &PyDict,
    max_period: usize,
    initial_alphabets: Vec<String>,
    steps: usize,
    restarts: usize,
    seed: u64,
    top_n: usize,
    guided: bool,
    guided_pool_size: usize,
) -> PyResult<PyObject> {
    if symbol_values.len() < 12 {
        return Err(PyValueError::new_err(
            "symbol_values must contain at least 12 A-Z values",
        ));
    }
    if symbol_values.iter().any(|v| *v >= 26) {
        return Err(PyValueError::new_err(
            "symbol_values must be A-Z values in 0..25",
        ));
    }

    let table = table_from_pydict(log_probs)?;
    let mut starts = Vec::new();
    if initial_alphabets.is_empty() {
        starts.push(*AZ);
    } else {
        for raw in initial_alphabets {
            starts.push(parse_alphabet(&raw)?);
        }
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut candidates = Vec::new();
    let max_period = max_period.max(1).min((symbol_values.len() / 4).max(1));
    let restart_count = restarts.max(1);
    for start in starts {
        for restart in 0..restart_count {
            let alpha = if restart == 0 {
                start
            } else {
                scramble_alphabet(start, &mut rng, 3 + restart * 2)
            };
            let values = keyed_indices(&symbol_values, &alpha);
            for period in 1..=max_period {
                candidates.push(anneal_period(
                    &symbol_values,
                    &values,
                    alpha,
                    period,
                    &table,
                    steps,
                    &mut rng,
                    restart,
                    guided,
                    guided_pool_size.max(1),
                ));
            }
        }
    }

    candidates.sort_by(|a, b| b.selection_score.total_cmp(&a.selection_score));
    let out = PyList::empty_bound(py);
    for candidate in candidates.iter().take(top_n.max(1)) {
        let d = PyDict::new_bound(py);
        d.set_item("variant", "keyed_vigenere")?;
        d.set_item("period", candidate.period)?;
        d.set_item("shifts", candidate.shifts.clone())?;
        d.set_item("key", key_string(&candidate.shifts, &candidate.alphabet))?;
        d.set_item("score", round5(candidate.score))?;
        d.set_item("selection_score", round5(candidate.selection_score))?;
        d.set_item("init_score", round5(candidate.init_score))?;
        d.set_item("plaintext", candidate.plaintext.clone())?;
        d.set_item(
            "preview",
            candidate.plaintext.chars().take(240).collect::<String>(),
        )?;
        let meta = PyDict::new_bound(py);
        meta.set_item("key_type", "PeriodicAlphabetKey")?;
        meta.set_item(
            "keyed_alphabet",
            String::from_utf8_lossy(&candidate.alphabet).to_string(),
        )?;
        meta.set_item(
            "initial_keyed_alphabet",
            String::from_utf8_lossy(&candidate.initial_alphabet).to_string(),
        )?;
        meta.set_item("initial_candidate_type", "rust_initial_alphabet")?;
        meta.set_item("restart", candidate.restart)?;
        meta.set_item(
            "mutation_search",
            if guided {
                "rust_guided_frequency_phase_swap_move_reverse"
            } else {
                "rust_swap_move_reverse"
            },
        )?;
        meta.set_item("guided", guided)?;
        meta.set_item("guided_pool_size", guided_pool_size.max(1))?;
        meta.set_item("guided_proposals", candidate.guided_proposals)?;
        meta.set_item("random_proposals", candidate.random_proposals)?;
        meta.set_item("score_model", "wordlist_quadgram")?;
        d.set_item("metadata", meta)?;
        out.append(d)?;
    }
    Ok(out.into())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn quagmire3_shotgun_search(
    py: Python<'_>,
    symbol_values: Vec<usize>,
    log_probs: &PyDict,
    keyword_lengths: Vec<usize>,
    cycleword_lengths: Vec<usize>,
    hillclimbs: usize,
    restarts: usize,
    seed: u64,
    top_n: usize,
    slip_probability: f64,
    backtrack_probability: f64,
    threads: usize,
    initial_keywords: Vec<String>,
) -> PyResult<PyObject> {
    if symbol_values.len() < 12 {
        return Err(PyValueError::new_err(
            "symbol_values must contain at least 12 A-Z values",
        ));
    }
    if symbol_values.iter().any(|v| *v >= 26) {
        return Err(PyValueError::new_err(
            "symbol_values must be A-Z values in 0..25",
        ));
    }
    let table = dense_table_from_pydict(log_probs, 4)?;
    let keyword_lengths = normalize_lengths(keyword_lengths, 7, 26);
    let cycleword_lengths = normalize_lengths(cycleword_lengths, 8, symbol_values.len().max(1));
    let restart_count = restarts.max(1);
    let top_n = top_n.max(1);
    let worker_count = if threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        threads.max(1)
    };

    let mut jobs = Vec::new();
    let mut job_index = 0usize;
    for &keyword_len in &keyword_lengths {
        for &period in &cycleword_lengths {
            for restart in 0..restart_count {
                jobs.push((job_index, keyword_len, period, restart));
                job_index += 1;
            }
        }
    }

    let explicit_starts = parse_keyword_starts(&initial_keywords)?;
    let pool = ThreadPoolBuilder::new()
        .num_threads(worker_count)
        .build()
        .map_err(|e| PyValueError::new_err(format!("failed to build rayon pool: {e}")))?;
    let mut candidates: Vec<QuagmireCandidate> = pool.install(|| {
        jobs.par_iter()
            .map(|(job_id, keyword_len, period, restart)| {
                let start = if !explicit_starts.is_empty() && *restart < explicit_starts.len() {
                    explicit_starts[*restart]
                } else {
                    let mut rng = StdRng::seed_from_u64(
                        seed ^ ((*job_id as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                    );
                    random_keyword_alphabet(*keyword_len, &mut rng)
                };
                let mut rng = StdRng::seed_from_u64(
                    seed ^ ((*job_id as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9))
                        ^ ((*period as u64).wrapping_mul(0x94D0_49BB_1331_11EB)),
                );
                quagmire_restart(
                    &symbol_values,
                    start,
                    *keyword_len,
                    *period,
                    *restart,
                    hillclimbs,
                    slip_probability.max(0.0),
                    backtrack_probability.max(0.0),
                    &table,
                    &mut rng,
                )
            })
            .collect()
    });

    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    let out = PyList::empty_bound(py);
    for candidate in candidates.iter().take(top_n) {
        let d = PyDict::new_bound(py);
        d.set_item("variant", "quag3")?;
        d.set_item("period", candidate.period)?;
        d.set_item("shifts", candidate.cycle_shifts.clone())?;
        d.set_item(
            "key",
            key_string(&candidate.cycle_shifts, &candidate.alphabet),
        )?;
        d.set_item("score", round5(candidate.score))?;
        d.set_item("selection_score", round5(candidate.score))?;
        d.set_item("plaintext", candidate.plaintext.clone())?;
        d.set_item(
            "preview",
            candidate.plaintext.chars().take(240).collect::<String>(),
        )?;
        let meta = PyDict::new_bound(py);
        meta.set_item("key_type", "QuagmireKey")?;
        meta.set_item("quagmire_type", "quag3")?;
        meta.set_item(
            "alphabet_keyword",
            keyword_prefix(&candidate.alphabet, candidate.keyword_len),
        )?;
        meta.set_item(
            "plaintext_alphabet",
            String::from_utf8_lossy(&candidate.alphabet).to_string(),
        )?;
        meta.set_item(
            "ciphertext_alphabet",
            String::from_utf8_lossy(&candidate.alphabet).to_string(),
        )?;
        meta.set_item(
            "cycleword",
            key_string(&candidate.cycle_shifts, &candidate.alphabet),
        )?;
        meta.set_item("cycleword_shifts", candidate.cycle_shifts.clone())?;
        meta.set_item("keyword_length", candidate.keyword_len)?;
        meta.set_item("restart", candidate.restart)?;
        meta.set_item("hillclimb_step", candidate.hillclimb_step)?;
        meta.set_item("accepted_mutations", candidate.accepted_mutations)?;
        meta.set_item("slips", candidate.slips)?;
        meta.set_item("backtracks", candidate.backtracks)?;
        meta.set_item("score_model", "rust_dense_quadgram_no_boundary")?;
        meta.set_item("mutation_search", "blake_style_keyword_restart_hillclimb")?;
        d.set_item("metadata", meta)?;
        out.append(d)?;
    }

    let result = PyDict::new_bound(py);
    result.set_item(
        "status",
        if candidates.is_empty() {
            "no_candidates"
        } else {
            "completed"
        },
    )?;
    result.set_item("solver", "quagmire3_shotgun_rust")?;
    result.set_item("seed", seed)?;
    result.set_item("threads", worker_count)?;
    result.set_item("restart_jobs", jobs.len())?;
    result.set_item("hillclimbs_per_restart", hillclimbs)?;
    result.set_item(
        "nominal_proposals",
        jobs.len().saturating_mul(hillclimbs.max(1)),
    )?;
    result.set_item("keyword_lengths", keyword_lengths)?;
    result.set_item("cycleword_lengths", cycleword_lengths)?;
    result.set_item("slip_probability", slip_probability.max(0.0))?;
    result.set_item("backtrack_probability", backtrack_probability.max(0.0))?;
    result.set_item("initial_keywords", initial_keywords)?;
    result.set_item("initial_keyword_count", explicit_starts.len())?;
    result.set_item(
        "attribution",
        "Quagmire III search strategy is inspired by Sam Blake's MIT-licensed polyalphabetic solver; this Rust kernel is a Decipher implementation.",
    )?;
    result.set_item("top_candidates", out)?;
    Ok(result.into())
}

#[pyfunction]
fn transform_structural_metrics_batch(
    py: Python<'_>,
    tokens: Vec<usize>,
    candidates: &PyList,
    threads: usize,
) -> PyResult<PyObject> {
    let mut parsed = Vec::with_capacity(candidates.len());
    for item in candidates.iter() {
        let dict = item.downcast::<PyDict>()?;
        parsed.push(parse_transform_candidate(dict)?);
    }
    let worker_count = if threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        threads.max(1)
    };
    let token_count = tokens.len();
    let pool = ThreadPoolBuilder::new()
        .num_threads(worker_count)
        .build()
        .map_err(|e| PyValueError::new_err(format!("failed to build rayon pool: {e}")))?;
    let results: Vec<FastTransformScore> = pool.install(|| {
        parsed
            .par_iter()
            .map(|candidate| score_transform_candidate_fast(&tokens, token_count, candidate))
            .collect()
    });

    let out = PyList::empty_bound(py);
    for result in results {
        let d = PyDict::new_bound(py);
        d.set_item("valid", result.valid)?;
        d.set_item("reason", result.reason)?;
        d.set_item("token_order_hash", result.token_order_hash)?;
        d.set_item("position_order_preview", result.position_order_preview)?;
        d.set_item("transformed_preview", result.transformed_preview)?;
        if let Some(metrics) = result.metrics {
            let m = PyDict::new_bound(py);
            m.set_item("repeated_bigram_rate", metrics.repeated_bigram_rate)?;
            m.set_item("repeated_trigram_rate", metrics.repeated_trigram_rate)?;
            m.set_item("alternation_rate", metrics.alternation_rate)?;
            m.set_item("symbol_ioc", metrics.symbol_ioc)?;
            m.set_item("position_nontriviality", metrics.position_nontriviality)?;
            m.set_item("position_adjacent_rate", metrics.position_adjacent_rate)?;
            m.set_item(
                "position_step_repeat_rate",
                metrics.position_step_repeat_rate,
            )?;
            m.set_item("periodic_redundancy", metrics.periodic_redundancy)?;
            m.set_item(
                "inverse_periodic_redundancy",
                metrics.inverse_periodic_redundancy,
            )?;
            m.set_item("best_period", metrics.best_period)?;
            m.set_item("inverse_best_period", metrics.inverse_best_period)?;
            m.set_item("grid_row_step_rate", metrics.grid_row_step_rate)?;
            m.set_item("grid_column_step_rate", metrics.grid_column_step_rate)?;
            m.set_item("periodic_structure_score", metrics.periodic_structure_score)?;
            m.set_item("matrix_rank_score", metrics.matrix_rank_score)?;
            d.set_item("metrics", m)?;
        } else {
            d.set_item("metrics", py.None())?;
        }
        out.append(d)?;
    }
    Ok(out.into())
}

fn parse_transform_candidate(raw: &PyDict) -> PyResult<FastTransformCandidate> {
    let pipeline_obj = raw
        .get_item("pipeline")?
        .ok_or_else(|| PyValueError::new_err("transform candidate missing pipeline"))?;
    let pipeline = parse_transform_pipeline(pipeline_obj)?;
    let (grid_columns, grid_rows) = if let Some(grid_obj) = raw.get_item("grid")? {
        if grid_obj.is_none() {
            (None, None)
        } else {
            let grid = grid_obj.downcast::<PyDict>()?;
            (
                py_optional_usize(grid, "columns")?,
                py_optional_usize(grid, "rows")?,
            )
        }
    } else {
        (None, None)
    };
    Ok(FastTransformCandidate {
        pipeline,
        grid_columns,
        grid_rows,
    })
}

fn parse_transform_pipeline(raw: &PyAny) -> PyResult<FastTransformPipeline> {
    if raw.is_none() {
        return Ok(FastTransformPipeline {
            columns: None,
            rows: None,
            steps: Vec::new(),
        });
    }
    if let Ok(list) = raw.downcast::<PyList>() {
        return Ok(FastTransformPipeline {
            columns: None,
            rows: None,
            steps: parse_transform_steps(list)?,
        });
    }
    let dict = raw.downcast::<PyDict>()?;
    let steps_obj = dict.get_item("steps")?.or_else(|| {
        dict.get_item("appliedCiphertextTransformers")
            .ok()
            .flatten()
    });
    let steps = if let Some(obj) = steps_obj {
        parse_transform_steps(obj.downcast::<PyList>()?)?
    } else {
        Vec::new()
    };
    Ok(FastTransformPipeline {
        columns: py_optional_usize(dict, "columns")?.or(py_optional_usize(dict, "grid_columns")?),
        rows: py_optional_usize(dict, "rows")?.or(py_optional_usize(dict, "grid_rows")?),
        steps,
    })
}

fn parse_transform_steps(raw: &PyList) -> PyResult<Vec<FastTransformStep>> {
    let mut out = Vec::with_capacity(raw.len());
    for item in raw.iter() {
        let dict = item.downcast::<PyDict>()?;
        let name_obj = dict
            .get_item("name")?
            .or_else(|| dict.get_item("transformerName").ok().flatten())
            .or_else(|| dict.get_item("transformer_name").ok().flatten())
            .ok_or_else(|| PyValueError::new_err("transform step missing name"))?;
        let name: String = name_obj.extract()?;
        let data = if let Some(data_obj) = dict.get_item("data")? {
            if data_obj.is_none() {
                HashMap::new()
            } else {
                parse_fast_data(data_obj.downcast::<PyDict>()?)?
            }
        } else {
            let mut data = HashMap::new();
            for (key, value) in dict.iter() {
                let key_string: String = key.extract()?;
                if matches!(
                    key_string.as_str(),
                    "name" | "transformerName" | "transformer_name"
                ) {
                    continue;
                }
                if let Some(parsed) = parse_fast_value(value)? {
                    data.insert(key_string, parsed);
                }
            }
            data
        };
        out.push(FastTransformStep { name, data });
    }
    Ok(out)
}

fn parse_fast_data(raw: &PyDict) -> PyResult<HashMap<String, FastValue>> {
    let mut out = HashMap::new();
    for (key, value) in raw.iter() {
        let key_string: String = key.extract()?;
        if let Some(parsed) = parse_fast_value(value)? {
            out.insert(key_string, parsed);
        }
    }
    Ok(out)
}

fn parse_fast_value(raw: &PyAny) -> PyResult<Option<FastValue>> {
    if raw.is_none() {
        return Ok(None);
    }
    if let Ok(value) = raw.extract::<i64>() {
        return Ok(Some(FastValue::Int(value)));
    }
    if let Ok(value) = raw.extract::<String>() {
        return Ok(Some(FastValue::Str(value)));
    }
    if let Ok(list) = raw.downcast::<PyList>() {
        let mut out = Vec::with_capacity(list.len());
        for item in list.iter() {
            out.push(item.extract::<usize>()?);
        }
        return Ok(Some(FastValue::IntList(out)));
    }
    Ok(None)
}

fn py_optional_usize(dict: &PyDict, key: &str) -> PyResult<Option<usize>> {
    Ok(match dict.get_item(key)? {
        Some(value) if !value.is_none() => Some(value.extract::<usize>()?),
        _ => None,
    })
}

fn fast_int(data: &HashMap<String, FastValue>, key: &str, default: i64) -> i64 {
    match data.get(key) {
        Some(FastValue::Int(value)) => *value,
        Some(FastValue::Str(value)) => value.parse::<i64>().unwrap_or(default),
        _ => default,
    }
}

fn fast_string(data: &HashMap<String, FastValue>, keys: &[&str], default: &str) -> String {
    for key in keys {
        if let Some(FastValue::Str(value)) = data.get(*key) {
            return value.clone();
        }
    }
    default.to_string()
}

fn fast_order_value(
    data: &HashMap<String, FastValue>,
    keys: &[&str],
    size: usize,
) -> PyResult<Vec<usize>> {
    for key in keys {
        if let Some(value) = data.get(*key) {
            return match value {
                FastValue::IntList(order) => {
                    if is_permutation(order, size) {
                        Ok(order.clone())
                    } else {
                        Err(PyValueError::new_err(
                            "explicit grid order is not a permutation",
                        ))
                    }
                }
                FastValue::Str(name) => grid_order(name, size),
                FastValue::Int(_) => grid_order("identity", size),
            };
        }
    }
    grid_order("identity", size)
}

fn score_transform_candidate_fast(
    source_tokens: &[usize],
    token_count: usize,
    candidate: &FastTransformCandidate,
) -> FastTransformScore {
    match apply_fast_pipeline(&(0..token_count).collect::<Vec<_>>(), &candidate.pipeline) {
        Ok(order) => {
            if order.len() != token_count {
                return invalid_transform_score(
                    format!("length_changed:{}", order.len()),
                    Some(token_hash(&order)),
                );
            }
            if !is_permutation(&order, token_count) {
                return invalid_transform_score(
                    "not_a_permutation".to_string(),
                    Some(token_hash(&order)),
                );
            }
            let metrics =
                fast_structural_metrics(&order, candidate.grid_columns, candidate.grid_rows);
            let transformed_preview = order
                .iter()
                .take(60)
                .map(|idx| source_tokens.get(*idx).copied().unwrap_or(0))
                .collect();
            FastTransformScore {
                valid: true,
                reason: "ok".to_string(),
                token_order_hash: Some(token_hash(&order)),
                position_order_preview: order.iter().take(60).copied().collect(),
                transformed_preview,
                metrics: Some(metrics),
            }
        }
        Err(err) => invalid_transform_score(err, None),
    }
}

fn invalid_transform_score(reason: String, token_order_hash: Option<String>) -> FastTransformScore {
    FastTransformScore {
        valid: false,
        reason,
        token_order_hash,
        position_order_preview: Vec::new(),
        transformed_preview: Vec::new(),
        metrics: None,
    }
}

fn apply_fast_pipeline(
    tokens: &[usize],
    pipeline: &FastTransformPipeline,
) -> Result<Vec<usize>, String> {
    let mut current = tokens.to_vec();
    let mut locked = vec![false; current.len()];
    for step in &pipeline.steps {
        let (new_tokens, new_locked) = apply_fast_step(&current, &locked, pipeline, step)?;
        current = new_tokens;
        locked = new_locked;
    }
    Ok(current)
}

fn apply_fast_step(
    tokens: &[usize],
    locked: &[bool],
    pipeline: &FastTransformPipeline,
    step: &FastTransformStep,
) -> Result<(Vec<usize>, Vec<bool>), String> {
    let name = canonical_transform_name(&step.name);
    match name.as_str() {
        "reverse" => {
            let (start, end) = transform_range(&step.data, tokens.len());
            let mut new_tokens = tokens.to_vec();
            let mut new_locked = locked.to_vec();
            if start <= end {
                new_tokens[start..=end].reverse();
                new_locked[start..=end].reverse();
            }
            Ok((new_tokens, new_locked))
        }
        "shiftcharactersleft" => {
            let (start, end) = transform_range(&step.data, tokens.len());
            Ok(shift_range(tokens, locked, start, end, -1))
        }
        "shiftcharactersright" => {
            let (start, end) = transform_range(&step.data, tokens.len());
            Ok(shift_range(tokens, locked, start, end, 1))
        }
        "lockcharacters" => {
            let (start, end) = transform_range(&step.data, tokens.len());
            let mut new_locked = locked.to_vec();
            for item in new_locked.iter_mut().take(end + 1).skip(start) {
                *item = true;
            }
            Ok((tokens.to_vec(), new_locked))
        }
        "ndownmacross" => n_down_m_across_fast(tokens, locked, pipeline, &step.data),
        "transposition" => columnar_transposition_fast(tokens, locked, &step.data, false),
        "unwraptransposition" => columnar_transposition_fast(tokens, locked, &step.data, true),
        "routeread" => route_read_fast(tokens, locked, pipeline, &step.data),
        "splitgridroute" => split_grid_route_fast(tokens, locked, pipeline, &step.data),
        "gridpermute" => grid_permute_fast(tokens, locked, pipeline, &step.data),
        _ => Err(format!("unsupported ciphertext transformer: {}", step.name)),
    }
}

fn canonical_transform_name(name: &str) -> String {
    name.chars()
        .filter(|ch| ch.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn transform_range(data: &HashMap<String, FastValue>, length: usize) -> (usize, usize) {
    if length == 0 {
        return (0, usize::MAX);
    }
    let start = fast_int(data, "rangeStart", 0).max(0) as usize;
    let end_default = length.saturating_sub(1) as i64;
    let end = fast_int(data, "rangeEnd", end_default)
        .max(0)
        .min(end_default) as usize;
    (start.min(length - 1), end)
}

fn shift_range(
    tokens: &[usize],
    locked: &[bool],
    start: usize,
    end: usize,
    direction: i32,
) -> (Vec<usize>, Vec<bool>) {
    let mut new_tokens = tokens.to_vec();
    let mut new_locked = locked.to_vec();
    if start > end || end >= tokens.len() {
        return (new_tokens, new_locked);
    }
    if direction < 0 {
        for i in start..end {
            new_tokens[i] = tokens[i + 1];
            new_locked[i] = locked[i + 1];
        }
        new_tokens[end] = tokens[start];
        new_locked[end] = locked[start];
    } else {
        new_tokens[start] = tokens[end];
        new_locked[start] = locked[end];
        for i in (start + 1)..=end {
            new_tokens[i] = tokens[i - 1];
            new_locked[i] = locked[i - 1];
        }
    }
    (new_tokens, new_locked)
}

fn n_down_m_across_fast(
    tokens: &[usize],
    locked: &[bool],
    pipeline: &FastTransformPipeline,
    data: &HashMap<String, FastValue>,
) -> Result<(Vec<usize>, Vec<bool>), String> {
    let columns = pipeline
        .columns
        .ok_or_else(|| "NDownMAcross requires pipeline columns".to_string())?;
    if columns == 0 {
        return Err("NDownMAcross requires pipeline columns".to_string());
    }
    let rows = pipeline.rows.unwrap_or(tokens.len() / columns);
    if rows == 0 {
        return Ok((tokens.to_vec(), locked.to_vec()));
    }
    let row_start = fast_int(data, "rangeStart", 0).max(0) as usize;
    let row_end = fast_int(data, "rangeEnd", rows.saturating_sub(1) as i64)
        .max(0)
        .min(rows.saturating_sub(1) as i64) as usize;
    if row_start > row_end {
        return Ok((tokens.to_vec(), locked.to_vec()));
    }
    let down = fast_int(data, "down", 1) as usize;
    let across = fast_int(data, "across", 1) as usize;
    let char_start = row_start * columns;
    let char_end = ((row_end + 1) * columns).min(tokens.len());
    if char_start >= char_end {
        return Ok((tokens.to_vec(), locked.to_vec()));
    }
    let mut new_tokens = tokens.to_vec();
    let mut new_locked = locked.to_vec();
    let mut output = char_start;
    let mut cursor = char_start;
    let mut append_later = Vec::new();
    let offset = down * columns + across;
    for _ in char_start..char_end {
        if !locked[cursor] {
            new_tokens[output] = tokens[cursor];
            new_locked[output] = locked[cursor];
            output += 1;
        } else {
            append_later.push(cursor);
        }
        let previous = cursor;
        cursor += offset;
        if cursor.saturating_sub(across) >= char_end {
            cursor = char_start + (cursor % columns);
        } else if (previous % columns) > (cursor % columns) {
            cursor = cursor.saturating_sub(columns);
        }
        if cursor >= tokens.len() {
            return Err("not_a_permutation".to_string());
        }
    }
    let tail_start = char_end.saturating_sub(append_later.len());
    for (i, source_index) in append_later.iter().enumerate() {
        new_tokens[tail_start + i] = tokens[*source_index];
        new_locked[tail_start + i] = locked[*source_index];
    }
    Ok((new_tokens, new_locked))
}

fn columnar_transposition_fast(
    tokens: &[usize],
    locked: &[bool],
    data: &HashMap<String, FastValue>,
    unwrap: bool,
) -> Result<(Vec<usize>, Vec<bool>), String> {
    let key = fast_string(data, &["key", "argument"], "");
    let indices = indices_for_key(&key);
    if indices.len() < 2 || indices.len() >= tokens.len() {
        return Err(
            "transposition key length must be greater than 1 and less than token count".to_string(),
        );
    }
    let rows = tokens.len() / indices.len();
    let usable = rows * indices.len();
    let mut new_tokens = tokens.to_vec();
    let mut new_locked = locked.to_vec();
    let mut out = 0usize;
    for i in 0..indices.len() {
        let column_index = indices
            .iter()
            .position(|value| *value == i)
            .ok_or_else(|| "invalid transposition key".to_string())?;
        for row in 0..rows {
            let mut source = row * indices.len() + column_index;
            let target;
            if unwrap {
                target = source;
                source = out;
            } else {
                target = out;
            }
            new_tokens[target] = tokens[source];
            new_locked[target] = locked[source];
            out += 1;
        }
    }
    if usable < tokens.len() {
        new_tokens[usable..].copy_from_slice(&tokens[usable..]);
        new_locked[usable..].copy_from_slice(&locked[usable..]);
    }
    Ok((new_tokens, new_locked))
}

fn indices_for_key(key: &str) -> Vec<usize> {
    let lower: Vec<u8> = key.bytes().map(|ch| ch.to_ascii_lowercase()).collect();
    let mut indices = vec![usize::MAX; lower.len()];
    let mut next_index = 0usize;
    for letter in b'a'..=b'z' {
        for (pos, ch) in lower.iter().enumerate() {
            if *ch == letter {
                indices[pos] = next_index;
                next_index += 1;
            }
        }
    }
    indices
}

fn route_read_fast(
    tokens: &[usize],
    locked: &[bool],
    pipeline: &FastTransformPipeline,
    data: &HashMap<String, FastValue>,
) -> Result<(Vec<usize>, Vec<bool>), String> {
    let columns = fast_int(data, "columns", pipeline.columns.unwrap_or(0) as i64) as usize;
    let mut rows = fast_int(data, "rows", pipeline.rows.unwrap_or(0) as i64) as usize;
    if columns == 0 {
        return Err("RouteRead requires columns".to_string());
    }
    if rows == 0 {
        rows = tokens.len() / columns;
    }
    let usable = (rows * columns).min(tokens.len());
    if rows == 0 || usable == 0 {
        return Ok((tokens.to_vec(), locked.to_vec()));
    }
    let route = fast_string(data, &["route"], "columns_down").to_lowercase();
    let positions = if route == "offset_chain" {
        offset_chain_positions(rows, columns, fast_int(data, "step", 1) as usize)
    } else if route == "rows_progressive_shift" || route == "columns_progressive_shift" {
        progressive_shift_positions(rows, columns, &route, fast_int(data, "shift", 1) as usize)
    } else {
        route_positions(rows, columns, &route)?
    };
    if positions.len() != rows * columns {
        return Err(format!("route {route:?} did not cover the grid"));
    }
    let order: Vec<usize> = positions
        .into_iter()
        .map(|(row, col)| row * columns + col)
        .filter(|idx| *idx < usable)
        .collect();
    if !is_permutation(&order, usable) {
        return Err(format!(
            "route {route:?} did not produce a grid permutation"
        ));
    }
    let mut new_tokens: Vec<usize> = order.iter().map(|idx| tokens[*idx]).collect();
    new_tokens.extend_from_slice(&tokens[usable..]);
    let mut new_locked: Vec<bool> = order.iter().map(|idx| locked[*idx]).collect();
    new_locked.extend_from_slice(&locked[usable..]);
    Ok((new_tokens, new_locked))
}

fn split_grid_route_fast(
    tokens: &[usize],
    locked: &[bool],
    pipeline: &FastTransformPipeline,
    data: &HashMap<String, FastValue>,
) -> Result<(Vec<usize>, Vec<bool>), String> {
    let columns = fast_int(data, "columns", pipeline.columns.unwrap_or(0) as i64) as usize;
    let mut rows = fast_int(data, "rows", pipeline.rows.unwrap_or(0) as i64) as usize;
    if columns <= 1 {
        return Err("SplitGridRoute requires columns > 1".to_string());
    }
    if rows == 0 {
        rows = tokens.len() / columns;
    }
    let usable = (rows * columns).min(tokens.len());
    if rows <= 1 || usable == 0 {
        return Ok((tokens.to_vec(), locked.to_vec()));
    }
    let orientation = fast_string(data, &["orientation"], "horizontal").to_lowercase();
    let default_split = if orientation == "horizontal" {
        rows / 2
    } else {
        columns / 2
    };
    let split = fast_int(data, "split", default_split as i64) as usize;
    let first_route = fast_string(data, &["firstRoute", "first_route"], "rows").to_lowercase();
    let second_route = fast_string(data, &["secondRoute", "second_route"], "rows").to_lowercase();
    let region_order = fast_string(data, &["regionOrder", "region_order"], "normal").to_lowercase();
    let (first, second) = if orientation == "horizontal" {
        if split == 0 || split >= rows {
            return Err("horizontal SplitGridRoute split must be inside row range".to_string());
        }
        (
            subgrid_order(0, 0, split, columns, columns, &first_route)?,
            subgrid_order(split, 0, rows - split, columns, columns, &second_route)?,
        )
    } else if orientation == "vertical" {
        if split == 0 || split >= columns {
            return Err("vertical SplitGridRoute split must be inside column range".to_string());
        }
        (
            subgrid_order(0, 0, rows, split, columns, &first_route)?,
            subgrid_order(0, split, rows, columns - split, columns, &second_route)?,
        )
    } else {
        return Err("SplitGridRoute orientation must be horizontal or vertical".to_string());
    };
    let mut order = if region_order == "swap" {
        [second, first].concat()
    } else {
        [first, second].concat()
    };
    order.retain(|idx| *idx < usable);
    if !is_permutation(&order, usable) {
        return Err("SplitGridRoute did not produce a grid permutation".to_string());
    }
    let mut new_tokens: Vec<usize> = order.iter().map(|idx| tokens[*idx]).collect();
    new_tokens.extend_from_slice(&tokens[usable..]);
    let mut new_locked: Vec<bool> = order.iter().map(|idx| locked[*idx]).collect();
    new_locked.extend_from_slice(&locked[usable..]);
    Ok((new_tokens, new_locked))
}

fn grid_permute_fast(
    tokens: &[usize],
    locked: &[bool],
    pipeline: &FastTransformPipeline,
    data: &HashMap<String, FastValue>,
) -> Result<(Vec<usize>, Vec<bool>), String> {
    let columns = fast_int(data, "columns", pipeline.columns.unwrap_or(0) as i64) as usize;
    let mut rows = fast_int(data, "rows", pipeline.rows.unwrap_or(0) as i64) as usize;
    if columns <= 1 {
        return Err("GridPermute requires columns > 1".to_string());
    }
    if rows == 0 {
        rows = tokens.len() / columns;
    }
    let usable = (rows * columns).min(tokens.len());
    if rows <= 1 || usable == 0 {
        return Ok((tokens.to_vec(), locked.to_vec()));
    }
    let row_order =
        fast_order_value(data, &["rowOrder", "row_order"], rows).map_err(|e| e.to_string())?;
    let column_order = fast_order_value(data, &["columnOrder", "column_order"], columns)
        .map_err(|e| e.to_string())?;
    let mut order = Vec::with_capacity(usable);
    for row in row_order {
        for col in &column_order {
            let idx = row * columns + *col;
            if idx < usable {
                order.push(idx);
            }
        }
    }
    if !is_permutation(&order, usable) {
        return Err("GridPermute did not produce a grid permutation".to_string());
    }
    let mut new_tokens: Vec<usize> = order.iter().map(|idx| tokens[*idx]).collect();
    new_tokens.extend_from_slice(&tokens[usable..]);
    let mut new_locked: Vec<bool> = order.iter().map(|idx| locked[*idx]).collect();
    new_locked.extend_from_slice(&locked[usable..]);
    Ok((new_tokens, new_locked))
}

fn route_positions(
    rows: usize,
    columns: usize,
    route: &str,
) -> Result<Vec<(usize, usize)>, String> {
    match route {
        "rows" | "rows_right" | "rows_ltr" => Ok((0..rows)
            .flat_map(|r| (0..columns).map(move |c| (r, c)))
            .collect()),
        "rows_reverse" | "rows_left" | "rows_rtl" => Ok((0..rows)
            .flat_map(|r| (0..columns).rev().map(move |c| (r, c)))
            .collect()),
        "rows_boustrophedon" | "boustrophedon_rows" => Ok((0..rows)
            .flat_map(|r| {
                let cols: Vec<usize> = if r % 2 == 0 {
                    (0..columns).collect()
                } else {
                    (0..columns).rev().collect()
                };
                cols.into_iter().map(move |c| (r, c))
            })
            .collect()),
        "columns" | "columns_down" | "columns_ttb" => Ok((0..columns)
            .flat_map(|c| (0..rows).map(move |r| (r, c)))
            .collect()),
        "columns_up" | "columns_btt" => Ok((0..columns)
            .flat_map(|c| (0..rows).rev().map(move |r| (r, c)))
            .collect()),
        "columns_boustrophedon" | "boustrophedon_columns" => Ok((0..columns)
            .flat_map(|c| {
                let row_values: Vec<usize> = if c % 2 == 0 {
                    (0..rows).collect()
                } else {
                    (0..rows).rev().collect()
                };
                row_values.into_iter().map(move |r| (r, c))
            })
            .collect()),
        "spiral_clockwise" => Ok(spiral_positions(rows, columns, true)),
        "spiral_counterclockwise" => Ok(spiral_positions(rows, columns, false)),
        "diagonal_down_right" => Ok(diagonal_positions(rows, columns, true, true)),
        "diagonal_down_left" => Ok(diagonal_positions(rows, columns, true, false)),
        "diagonal_up_right" => Ok(diagonal_positions(rows, columns, false, true)),
        "diagonal_up_left" => Ok(diagonal_positions(rows, columns, false, false)),
        "diagonal_zigzag_down_right" => Ok(diagonal_zigzag_positions(rows, columns, true, true)),
        "diagonal_zigzag_down_left" => Ok(diagonal_zigzag_positions(rows, columns, true, false)),
        "checkerboard_even_odd" => Ok(checkerboard_positions(rows, columns, true)),
        "checkerboard_odd_even" => Ok(checkerboard_positions(rows, columns, false)),
        "row_column_interleave" => Ok(interleave_orders(
            &route_positions(rows, columns, "rows")?,
            &route_positions(rows, columns, "columns_down")?,
        )),
        "column_row_interleave" => Ok(interleave_orders(
            &route_positions(rows, columns, "columns_down")?,
            &route_positions(rows, columns, "rows")?,
        )),
        _ => Err(format!("unsupported RouteRead route: {route}")),
    }
}

fn spiral_positions(rows: usize, columns: usize, clockwise: bool) -> Vec<(usize, usize)> {
    if rows == 0 || columns == 0 {
        return Vec::new();
    }
    let mut top = 0isize;
    let mut bottom = rows as isize - 1;
    let mut left = 0isize;
    let mut right = columns as isize - 1;
    let mut out = Vec::with_capacity(rows * columns);
    while top <= bottom && left <= right {
        if clockwise {
            for c in left..=right {
                out.push((top as usize, c as usize));
            }
            top += 1;
            for r in top..=bottom {
                out.push((r as usize, right as usize));
            }
            right -= 1;
            if top <= bottom {
                for c in (left..=right).rev() {
                    out.push((bottom as usize, c as usize));
                }
                bottom -= 1;
            }
            if left <= right {
                for r in (top..=bottom).rev() {
                    out.push((r as usize, left as usize));
                }
                left += 1;
            }
        } else {
            for r in top..=bottom {
                out.push((r as usize, left as usize));
            }
            left += 1;
            for c in left..=right {
                out.push((bottom as usize, c as usize));
            }
            bottom -= 1;
            if left <= right {
                for r in (top..=bottom).rev() {
                    out.push((r as usize, right as usize));
                }
                right -= 1;
            }
            if top <= bottom {
                for c in (left..=right).rev() {
                    out.push((top as usize, c as usize));
                }
                top += 1;
            }
        }
    }
    out
}

fn diagonal_positions(rows: usize, columns: usize, down: bool, right: bool) -> Vec<(usize, usize)> {
    let row_order: Vec<usize> = if down {
        (0..rows).collect()
    } else {
        (0..rows).rev().collect()
    };
    let col_order: Vec<usize> = if right {
        (0..columns).collect()
    } else {
        (0..columns).rev().collect()
    };
    let mut row_rank = HashMap::new();
    let mut col_rank = HashMap::new();
    for (i, row) in row_order.iter().enumerate() {
        row_rank.insert(*row, i);
    }
    for (i, col) in col_order.iter().enumerate() {
        col_rank.insert(*col, i);
    }
    let mut out = Vec::with_capacity(rows * columns);
    for diagonal in 0..(rows + columns).saturating_sub(1) {
        for row in &row_order {
            for col in &col_order {
                if row_rank[row] + col_rank[col] == diagonal {
                    out.push((*row, *col));
                }
            }
        }
    }
    out
}

fn diagonal_zigzag_positions(
    rows: usize,
    columns: usize,
    down: bool,
    right: bool,
) -> Vec<(usize, usize)> {
    let row_order: Vec<usize> = if down {
        (0..rows).collect()
    } else {
        (0..rows).rev().collect()
    };
    let col_order: Vec<usize> = if right {
        (0..columns).collect()
    } else {
        (0..columns).rev().collect()
    };
    let mut row_rank = HashMap::new();
    let mut col_rank = HashMap::new();
    for (i, row) in row_order.iter().enumerate() {
        row_rank.insert(*row, i);
    }
    for (i, col) in col_order.iter().enumerate() {
        col_rank.insert(*col, i);
    }
    let mut out = Vec::with_capacity(rows * columns);
    for diagonal in 0..(rows + columns).saturating_sub(1) {
        let mut cells = Vec::new();
        for row in &row_order {
            for col in &col_order {
                if row_rank[row] + col_rank[col] == diagonal {
                    cells.push((*row, *col));
                }
            }
        }
        if diagonal % 2 == 1 {
            cells.reverse();
        }
        out.extend(cells);
    }
    out
}

fn checkerboard_positions(rows: usize, columns: usize, even_first: bool) -> Vec<(usize, usize)> {
    let parities = if even_first {
        [0usize, 1usize]
    } else {
        [1usize, 0usize]
    };
    let mut out = Vec::with_capacity(rows * columns);
    for parity in parities {
        for r in 0..rows {
            for c in 0..columns {
                if (r + c) % 2 == parity {
                    out.push((r, c));
                }
            }
        }
    }
    out
}

fn interleave_orders(first: &[(usize, usize)], second: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(first.len());
    let mut seen = std::collections::HashSet::new();
    let max_len = first.len().max(second.len());
    for i in 0..max_len {
        if let Some(value) = first.get(i) {
            if seen.insert(*value) {
                out.push(*value);
            }
        }
        if let Some(value) = second.get(i) {
            if seen.insert(*value) {
                out.push(*value);
            }
        }
    }
    out
}

fn offset_chain_positions(rows: usize, columns: usize, step: usize) -> Vec<(usize, usize)> {
    let size = rows * columns;
    if size == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(size);
    let mut seen = vec![false; size];
    let mut index = 0usize;
    for _ in 0..size {
        if seen[index] {
            break;
        }
        seen[index] = true;
        out.push((index / columns, index % columns));
        index = (index + step) % size;
    }
    for idx in 0..size {
        if !seen[idx] {
            out.push((idx / columns, idx % columns));
        }
    }
    out
}

fn progressive_shift_positions(
    rows: usize,
    columns: usize,
    route: &str,
    shift: usize,
) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(rows * columns);
    if route == "rows_progressive_shift" {
        for row in 0..rows {
            for offset in 0..columns {
                let col = (offset + row * shift) % columns;
                out.push((row, col));
            }
        }
    } else {
        for col in 0..columns {
            for offset in 0..rows {
                let row = (offset + col * shift) % rows;
                out.push((row, col));
            }
        }
    }
    out
}

fn subgrid_order(
    row_offset: usize,
    col_offset: usize,
    rows: usize,
    columns: usize,
    full_columns: usize,
    route: &str,
) -> Result<Vec<usize>, String> {
    let positions = route_positions(rows, columns, route)?;
    Ok(positions
        .into_iter()
        .map(|(row, col)| (row_offset + row) * full_columns + col_offset + col)
        .collect())
}

fn grid_order(raw: &str, size: usize) -> PyResult<Vec<usize>> {
    match raw.to_lowercase().as_str() {
        "identity" | "normal" | "rows" | "columns" => Ok((0..size).collect()),
        "reverse" | "reversed" => Ok((0..size).rev().collect()),
        "even_odd" => Ok((0..size).step_by(2).chain((1..size).step_by(2)).collect()),
        "odd_even" => Ok((1..size).step_by(2).chain((0..size).step_by(2)).collect()),
        "outside_in" => Ok(outside_in_order(size)),
        "inside_out" => {
            let mut out = outside_in_order(size);
            out.reverse();
            Ok(out)
        }
        other => Err(PyValueError::new_err(format!(
            "unsupported grid order: {other}"
        ))),
    }
}

fn outside_in_order(size: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(size);
    if size == 0 {
        return out;
    }
    let mut left = 0usize;
    let mut right = size - 1;
    while left <= right {
        out.push(left);
        if left != right {
            out.push(right);
        }
        left += 1;
        if right == 0 {
            break;
        }
        right -= 1;
    }
    out
}

fn is_permutation(order: &[usize], size: usize) -> bool {
    if order.len() != size {
        return false;
    }
    let mut seen = vec![false; size];
    for value in order {
        if *value >= size || seen[*value] {
            return false;
        }
        seen[*value] = true;
    }
    true
}

fn token_hash(tokens: &[usize]) -> String {
    let mut hasher = Sha1::new();
    for value in tokens {
        hasher.update((*value as u32).to_ne_bytes());
    }
    let digest = hasher.finalize();
    let hex = format!("{:x}", digest);
    hex[..16].to_string()
}

fn fast_structural_metrics(
    position_order: &[usize],
    columns: Option<usize>,
    rows: Option<usize>,
) -> FastTransformMetrics {
    if position_order.len() < 2 {
        return zero_transform_metrics();
    }
    let n = position_order.len();
    let fixed = position_order
        .iter()
        .enumerate()
        .filter(|(i, value)| *i == **value)
        .count();
    let nontriviality = 1.0 - (fixed as f64 / n as f64);
    let deltas: Vec<i64> = position_order
        .windows(2)
        .map(|w| w[1] as i64 - w[0] as i64)
        .collect();
    let denom = deltas.len().max(1) as f64;
    let adjacent_rate = deltas.iter().filter(|d| d.abs() == 1).count() as f64 / denom;
    let step_repeat_rate = repeat_scalar_rate_i64(&deltas);
    let column_step_rate = if let Some(cols) = columns {
        if cols > 1 {
            deltas.iter().filter(|d| d.abs() == cols as i64).count() as f64 / denom
        } else {
            0.0
        }
    } else {
        0.0
    };
    let periods = periodic_probe_periods(deltas.len(), columns, rows);
    let (periodic, best_period) = periodic_profile_i64(&deltas, &periods);
    let periodic_structure = nontriviality * periodic;
    let matrix_rank_score = nontriviality
        * (periodic * 0.45
            + step_repeat_rate * 0.25
            + adjacent_rate.max(column_step_rate) * 0.2
            + (1.0 - adjacent_rate.min(1.0)) * 0.1);
    FastTransformMetrics {
        repeated_bigram_rate: 0.0,
        repeated_trigram_rate: 0.0,
        alternation_rate: 0.0,
        symbol_ioc: 0.0,
        position_nontriviality: nontriviality,
        position_adjacent_rate: adjacent_rate,
        position_step_repeat_rate: step_repeat_rate,
        periodic_redundancy: periodic,
        inverse_periodic_redundancy: 0.0,
        best_period,
        inverse_best_period: 0.0,
        grid_row_step_rate: if columns.is_some() {
            adjacent_rate
        } else {
            0.0
        },
        grid_column_step_rate: column_step_rate,
        periodic_structure_score: periodic_structure,
        matrix_rank_score,
    }
}

fn zero_transform_metrics() -> FastTransformMetrics {
    FastTransformMetrics {
        repeated_bigram_rate: 0.0,
        repeated_trigram_rate: 0.0,
        alternation_rate: 0.0,
        symbol_ioc: 0.0,
        position_nontriviality: 0.0,
        position_adjacent_rate: 0.0,
        position_step_repeat_rate: 0.0,
        periodic_redundancy: 0.0,
        inverse_periodic_redundancy: 0.0,
        best_period: 0.0,
        inverse_best_period: 0.0,
        grid_row_step_rate: 0.0,
        grid_column_step_rate: 0.0,
        periodic_structure_score: 0.0,
        matrix_rank_score: 0.0,
    }
}

fn repeat_scalar_rate_i64(items: &[i64]) -> f64 {
    if items.is_empty() {
        return 0.0;
    }
    let mut counts = HashMap::<i64, usize>::new();
    for item in items {
        *counts.entry(*item).or_insert(0) += 1;
    }
    let repeated: usize = counts.values().filter(|n| **n > 1).sum();
    repeated as f64 / items.len() as f64
}

fn periodic_probe_periods(
    length: usize,
    columns: Option<usize>,
    rows: Option<usize>,
) -> Vec<usize> {
    let max_period = 40usize.min(length / 2);
    if max_period == 0 {
        return Vec::new();
    }
    let mut raw = vec![1, 2, 3, 4, 5, 8, 10, 12, 16, 20, 24, 32, 40];
    if let Some(cols) = columns {
        if cols > 1 {
            raw.extend([cols.saturating_sub(1), cols, cols + 1, cols * 2]);
        }
    }
    if let Some(row_count) = rows {
        if row_count > 1 {
            raw.extend([
                row_count.saturating_sub(1),
                row_count,
                row_count + 1,
                row_count * 2,
            ]);
        }
    }
    let mut out = Vec::new();
    for period in raw {
        if (1..=max_period).contains(&period) && !out.contains(&period) {
            out.push(period);
        }
    }
    out
}

fn periodic_profile_i64(values: &[i64], periods: &[usize]) -> (f64, f64) {
    if values.len() < 4 {
        return (0.0, 0.0);
    }
    let mut best = 0.0;
    let mut best_period = 0usize;
    for period in periods {
        if *period == 0 || *period >= values.len() {
            continue;
        }
        let denom = values.len() - period;
        let hits = (0..denom)
            .filter(|i| values[*i] == values[*i + period])
            .count();
        let value = if denom > 0 {
            hits as f64 / denom as f64
        } else {
            0.0
        };
        if value > best {
            best = value;
            best_period = *period;
        }
    }
    (best, best_period as f64)
}

fn table_from_pydict(log_probs: &PyDict) -> PyResult<NGramTable> {
    let mut probs = HashMap::new();
    let mut floor = -10.0_f64;
    for (key, value) in log_probs.iter() {
        let k: String = key.extract()?;
        let v: f64 = value.extract()?;
        if k == "_floor" {
            floor = v;
        } else {
            probs.insert(k.into_bytes(), v);
        }
    }
    Ok(NGramTable { probs, floor })
}

fn dense_table_from_pydict(log_probs: &PyDict, n: usize) -> PyResult<DenseNGramTable> {
    if n == 0 {
        return Err(PyValueError::new_err("n must be positive"));
    }
    let size = 26usize.pow(n as u32);
    let mut floor = -10.0_f64;
    for (key, value) in log_probs.iter() {
        let k: String = key.extract()?;
        if k == "_floor" {
            floor = value.extract()?;
            break;
        }
    }
    let mut probs = vec![floor; size];
    for (key, value) in log_probs.iter() {
        let k: String = key.extract()?;
        if k.len() != n || !k.bytes().all(|c| c.is_ascii_uppercase()) {
            continue;
        }
        let mut index = 0usize;
        for ch in k.bytes() {
            index = index * 26 + (ch - b'A') as usize;
        }
        probs[index] = value.extract()?;
    }
    Ok(DenseNGramTable { n, probs, floor })
}

fn dense_score_indices(text: &[usize], table: &DenseNGramTable) -> f64 {
    if table.n == 0 || text.len() < table.n {
        return f64::NEG_INFINITY;
    }
    let mut total = 0.0;
    let count = text.len() - table.n + 1;
    for window in text.windows(table.n) {
        let mut index = 0usize;
        for &v in window {
            index = index * 26 + v;
        }
        total += table.probs.get(index).copied().unwrap_or(table.floor);
    }
    total / count as f64
}

fn clean_text_bytes(text: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len());
    let mut prev_space = true;
    for &raw in text {
        let c = raw.to_ascii_uppercase();
        if c.is_ascii_alphabetic() {
            out.push(c);
            prev_space = false;
        } else if !prev_space {
            out.push(b' ');
            prev_space = true;
        }
    }
    while out.last() == Some(&b' ') {
        out.pop();
    }
    out
}

fn normalized_score(text: &[u8], table: &NGramTable, n: usize) -> f64 {
    let clean = clean_text_bytes(text);
    if clean.len() < n || n == 0 {
        return f64::NEG_INFINITY;
    }
    let mut padded = vec![b' '; n - 1];
    padded.extend_from_slice(&clean);
    padded.extend(std::iter::repeat(b' ').take(n - 1));
    let count = padded.len() - n + 1;
    if count == 0 {
        return f64::NEG_INFINITY;
    }
    let mut total = 0.0;
    for gram in padded.windows(n) {
        total += table.probs.get(gram).copied().unwrap_or(table.floor);
    }
    total / count as f64
}

fn parse_alphabet(raw: &str) -> PyResult<[u8; 26]> {
    let letters: Vec<u8> = raw
        .bytes()
        .filter(|c| c.is_ascii_alphabetic())
        .map(|c| c.to_ascii_uppercase())
        .collect();
    if letters.len() != 26 {
        return Err(PyValueError::new_err(
            "initial alphabets must contain exactly 26 A-Z letters",
        ));
    }
    let mut seen = [false; 26];
    let mut out = [b'A'; 26];
    for (i, ch) in letters.iter().enumerate() {
        let idx = (ch - b'A') as usize;
        if seen[idx] {
            return Err(PyValueError::new_err(
                "initial alphabets must not repeat letters",
            ));
        }
        seen[idx] = true;
        out[i] = *ch;
    }
    Ok(out)
}

fn keyword_alphabet_from_keyword(raw: &str) -> PyResult<[u8; 26]> {
    let mut seen = [false; 26];
    let mut out = Vec::with_capacity(26);
    for ch in raw.bytes().chain(AZ.iter().copied()) {
        if !ch.is_ascii_alphabetic() {
            continue;
        }
        let upper = ch.to_ascii_uppercase();
        let idx = (upper - b'A') as usize;
        if !seen[idx] {
            seen[idx] = true;
            out.push(upper);
        }
    }
    if out.len() != 26 {
        return Err(PyValueError::new_err(
            "keyword alphabet must contain A-Z letters",
        ));
    }
    let mut alphabet = [b'A'; 26];
    alphabet.copy_from_slice(&out);
    Ok(alphabet)
}

fn parse_keyword_starts(keywords: &[String]) -> PyResult<Vec<[u8; 26]>> {
    let mut out = Vec::new();
    for keyword in keywords {
        let alpha = keyword_alphabet_from_keyword(keyword)?;
        if !out.iter().any(|existing| existing == &alpha) {
            out.push(alpha);
        }
    }
    Ok(out)
}

fn normalize_lengths(values: Vec<usize>, default_value: usize, max_value: usize) -> Vec<usize> {
    let mut out: Vec<usize> = values
        .into_iter()
        .map(|v| v.max(1).min(max_value.max(1)))
        .collect();
    if out.is_empty() {
        out.push(default_value.max(1).min(max_value.max(1)));
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn random_keyword_alphabet(keyword_len: usize, rng: &mut StdRng) -> [u8; 26] {
    let keyword_len = keyword_len.max(1).min(26);
    let mut letters = *AZ;
    letters.shuffle(rng);
    let mut out = [b'A'; 26];
    out[..keyword_len].copy_from_slice(&letters[..keyword_len]);
    let mut pos = keyword_len;
    for &ch in AZ.iter() {
        if !out[..keyword_len].contains(&ch) {
            out[pos] = ch;
            pos += 1;
        }
    }
    out
}

fn mutate_keyword_alphabet(
    mut alphabet: [u8; 26],
    keyword_len: usize,
    rng: &mut StdRng,
) -> [u8; 26] {
    let keyword_len = keyword_len.max(1).min(26);
    if rng.gen::<f64>() < 0.20 && keyword_len >= 2 {
        let i = rng.gen_range(0..keyword_len);
        let mut j = rng.gen_range(0..keyword_len);
        while i == j {
            j = rng.gen_range(0..keyword_len);
        }
        alphabet.swap(i, j);
        return alphabet;
    }

    let i = rng.gen_range(0..keyword_len);
    let j = rng.gen_range(keyword_len..26);
    let temp = alphabet[i];
    alphabet[i] = alphabet[j];
    for k in (j + 1)..26 {
        alphabet[k - 1] = alphabet[k];
    }
    for k in keyword_len..26 {
        if alphabet[k] > temp || k == 25 {
            for l in ((k + 1)..26).rev() {
                alphabet[l] = alphabet[l - 1];
            }
            alphabet[k] = temp;
            break;
        }
    }
    alphabet
}

fn keyword_prefix(alphabet: &[u8; 26], keyword_len: usize) -> String {
    String::from_utf8_lossy(&alphabet[..keyword_len.max(1).min(26)]).to_string()
}

fn keyed_indices(symbol_values: &[usize], alphabet: &[u8; 26]) -> Vec<usize> {
    let mut index = [0usize; 26];
    for (i, ch) in alphabet.iter().enumerate() {
        index[(ch - b'A') as usize] = i;
    }
    symbol_values.iter().map(|v| index[*v]).collect()
}

fn derive_quagmire_cycle_shifts(
    symbol_values: &[usize],
    alphabet: &[u8; 26],
    period: usize,
) -> Vec<usize> {
    let values = keyed_indices(symbol_values, alphabet);
    let mut shifts = Vec::with_capacity(period);
    for phase in 0..period {
        let stream: Vec<usize> = values.iter().skip(phase).step_by(period).copied().collect();
        let mut best_shift = 0usize;
        let mut best_dot = f64::NEG_INFINITY;
        for shift in 0..26 {
            let mut counts = [0usize; 26];
            for c in &stream {
                let plain_ch = alphabet[(c + 26 - shift) % 26];
                counts[(plain_ch - b'A') as usize] += 1;
            }
            let total = stream.len().max(1) as f64;
            let mut dot = 0.0;
            for i in 0..26 {
                dot += counts[i] as f64 * ENGLISH_FREQS[i];
            }
            dot /= total;
            if dot > best_dot {
                best_dot = dot;
                best_shift = shift;
            }
        }
        shifts.push(best_shift);
    }
    shifts
}

fn quagmire_plain_indices(
    symbol_values: &[usize],
    alphabet: &[u8; 26],
    shifts: &[usize],
) -> Vec<usize> {
    let values = keyed_indices(symbol_values, alphabet);
    values
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let shift = shifts[i % shifts.len()];
            (alphabet[(c + 26 - shift) % 26] - b'A') as usize
        })
        .collect()
}

fn indices_to_string(values: &[usize]) -> String {
    let bytes: Vec<u8> = values.iter().map(|v| b'A' + (*v as u8 % 26)).collect();
    String::from_utf8_lossy(&bytes).to_string()
}

#[allow(clippy::too_many_arguments)]
fn quagmire_restart(
    symbol_values: &[usize],
    start_alphabet: [u8; 26],
    keyword_len: usize,
    period: usize,
    restart: usize,
    hillclimbs: usize,
    slip_probability: f64,
    backtrack_probability: f64,
    table: &DenseNGramTable,
    rng: &mut StdRng,
) -> QuagmireCandidate {
    let mut current_alpha = start_alphabet;
    let mut current_shifts = derive_quagmire_cycle_shifts(symbol_values, &current_alpha, period);
    let mut current_plain_indices =
        quagmire_plain_indices(symbol_values, &current_alpha, &current_shifts);
    let mut current_score = dense_score_indices(&current_plain_indices, table);

    let mut best_alpha = current_alpha;
    let mut best_shifts = current_shifts.clone();
    let mut best_plain_indices = current_plain_indices.clone();
    let mut best_score = current_score;
    let mut best_step = 0usize;
    let mut accepted = 0usize;
    let mut slips = 0usize;
    let mut backtracks = 0usize;

    for step in 1..=hillclimbs {
        if rng.gen::<f64>() < backtrack_probability {
            current_alpha = best_alpha;
            current_shifts = best_shifts.clone();
            current_plain_indices = best_plain_indices.clone();
            current_score = best_score;
            backtracks += 1;
        }

        let trial_alpha = mutate_keyword_alphabet(current_alpha, keyword_len, rng);
        let trial_shifts = derive_quagmire_cycle_shifts(symbol_values, &trial_alpha, period);
        let trial_plain_indices =
            quagmire_plain_indices(symbol_values, &trial_alpha, &trial_shifts);
        let trial_score = dense_score_indices(&trial_plain_indices, table);

        if trial_score > current_score || rng.gen::<f64>() < slip_probability {
            if trial_score <= current_score {
                slips += 1;
            }
            accepted += 1;
            current_alpha = trial_alpha;
            current_shifts = trial_shifts;
            current_plain_indices = trial_plain_indices;
            current_score = trial_score;
        }

        if current_score > best_score {
            best_alpha = current_alpha;
            best_shifts = current_shifts.clone();
            best_plain_indices = current_plain_indices.clone();
            best_score = current_score;
            best_step = step;
        }
    }

    QuagmireCandidate {
        period,
        keyword_len,
        restart,
        hillclimb_step: best_step,
        alphabet: best_alpha,
        cycle_shifts: best_shifts,
        score: best_score,
        plaintext: indices_to_string(&best_plain_indices),
        accepted_mutations: accepted,
        slips,
        backtracks,
    }
}

#[allow(clippy::too_many_arguments)]
fn anneal_period(
    symbol_values: &[usize],
    values: &[usize],
    alphabet: [u8; 26],
    period: usize,
    table: &NGramTable,
    steps: usize,
    rng: &mut StdRng,
    restart: usize,
    guided: bool,
    guided_pool_size: usize,
) -> Candidate {
    let mut current_alpha = alphabet;
    let mut current_values = values.to_vec();
    let mut current_shifts = initial_keyed_shifts(&current_values, period, &current_alpha);
    let refined = refine_keyed_shifts(&current_values, &current_shifts, &current_alpha, table);
    current_shifts = refined.0;
    let mut current_plain = refined.1;
    let mut current_score = refined.2;
    let start_score = current_score;

    let mut best_alpha = current_alpha;
    let mut best_shifts = current_shifts.clone();
    let mut best_plain = current_plain.clone();
    let mut best_score = current_score;
    let mut guided_proposals = 0usize;
    let mut random_proposals = 0usize;

    for step in 0..steps {
        let guided_choices = if guided && rng.gen::<f64>() < 0.70 {
            guided_alphabet_mutations(
                symbol_values,
                &current_alpha,
                period,
                &current_shifts,
                rng,
                guided_pool_size,
            )
        } else {
            Vec::new()
        };
        let trial_alpha = if let Some(choice) = guided_choices.choose(rng) {
            guided_proposals += 1;
            *choice
        } else {
            random_proposals += 1;
            mutate_alphabet(current_alpha, rng)
        };
        let trial_values = keyed_indices(symbol_values, &trial_alpha);
        let trial_shifts = initial_keyed_shifts(&trial_values, period, &trial_alpha);
        let (trial_shifts, trial_plain, trial_score) =
            refine_keyed_shifts(&trial_values, &trial_shifts, &trial_alpha, table);
        let temp = 0.08 * (1.0 - (step as f64 / steps.max(1) as f64)) + 0.004;
        if trial_score > current_score
            || rng.gen::<f64>() < ((trial_score - current_score) / temp).exp()
        {
            current_alpha = trial_alpha;
            current_values = trial_values;
            current_shifts = trial_shifts;
            current_plain = trial_plain;
            current_score = trial_score;
            if trial_score > best_score {
                best_alpha = current_alpha;
                best_shifts = current_shifts.clone();
                best_plain = current_plain.clone();
                best_score = current_score;
            }
        }
    }

    Candidate {
        period,
        shifts: best_shifts,
        alphabet: best_alpha,
        initial_alphabet: alphabet,
        restart,
        score: best_score,
        selection_score: best_score - period_complexity_penalty(period),
        init_score: start_score,
        plaintext: best_plain,
        guided_proposals,
        random_proposals,
    }
}

fn initial_keyed_shifts(values: &[usize], period: usize, alphabet: &[u8; 26]) -> Vec<usize> {
    let mut shifts = Vec::with_capacity(period);
    for phase in 0..period {
        let stream: Vec<usize> = values.iter().skip(phase).step_by(period).copied().collect();
        let mut best_shift = 0;
        let mut best_score = f64::INFINITY;
        for shift in 0..26 {
            let plain = keyed_phase_plain_values(&stream, shift, alphabet);
            let score = chi2_english(&plain);
            if score < best_score {
                best_score = score;
                best_shift = shift;
            }
        }
        shifts.push(best_shift);
    }
    shifts
}

fn keyed_phase_plain_values(stream: &[usize], shift: usize, alphabet: &[u8; 26]) -> Vec<usize> {
    stream
        .iter()
        .map(|c| (alphabet[(c + 26 - shift) % 26] - b'A') as usize)
        .collect()
}

fn chi2_english(values: &[usize]) -> f64 {
    if values.is_empty() {
        return f64::INFINITY;
    }
    let mut counts = [0usize; 26];
    for v in values {
        counts[*v] += 1;
    }
    let n = values.len() as f64;
    let mut chi2 = 0.0;
    for i in 0..26 {
        let expected = ENGLISH_FREQS[i] * n;
        if expected > 0.0 {
            chi2 += (counts[i] as f64 - expected).powi(2) / expected;
        }
    }
    chi2
}

fn refine_keyed_shifts(
    values: &[usize],
    shifts: &[usize],
    alphabet: &[u8; 26],
    table: &NGramTable,
) -> (Vec<usize>, String, f64) {
    let mut best_shifts = shifts.to_vec();
    let mut best_plain = decode_keyed(values, &best_shifts, alphabet);
    let mut best_score = normalized_score(best_plain.as_bytes(), table, 4);
    let mut improved = true;
    let mut passes = 0;
    while improved && passes < 4 {
        improved = false;
        passes += 1;
        for phase in 0..best_shifts.len() {
            let mut phase_best_shift = best_shifts[phase];
            let mut phase_best_plain = best_plain.clone();
            let mut phase_best_score = best_score;
            for candidate_shift in 0..26 {
                if candidate_shift == best_shifts[phase] {
                    continue;
                }
                let mut trial = best_shifts.clone();
                trial[phase] = candidate_shift;
                let plain = decode_keyed(values, &trial, alphabet);
                let score = normalized_score(plain.as_bytes(), table, 4);
                if score > phase_best_score {
                    phase_best_shift = candidate_shift;
                    phase_best_plain = plain;
                    phase_best_score = score;
                }
            }
            if phase_best_shift != best_shifts[phase] {
                best_shifts[phase] = phase_best_shift;
                best_plain = phase_best_plain;
                best_score = phase_best_score;
                improved = true;
            }
        }
    }
    (best_shifts, best_plain, best_score)
}

fn decode_keyed(values: &[usize], shifts: &[usize], alphabet: &[u8; 26]) -> String {
    let mut out = Vec::with_capacity(values.len());
    for (i, c) in values.iter().enumerate() {
        let shift = shifts[i % shifts.len()];
        out.push(alphabet[(c + 26 - shift) % 26]);
    }
    String::from_utf8_lossy(&out).to_string()
}

fn guided_alphabet_mutations(
    symbol_values: &[usize],
    alphabet: &[u8; 26],
    period: usize,
    shifts: &[usize],
    rng: &mut StdRng,
    limit: usize,
) -> Vec<[u8; 26]> {
    let values = keyed_indices(symbol_values, alphabet);
    let mut candidates = Vec::new();
    candidates.extend(frequency_pressure_swaps(
        alphabet,
        &values,
        shifts,
        limit.max(4) / 2,
    ));
    candidates.extend(phase_common_letter_swaps(
        alphabet,
        &values,
        period,
        shifts,
        limit.max(4),
    ));
    candidates.sort();
    candidates.dedup();
    candidates.retain(|c| c != alphabet);
    candidates.shuffle(rng);
    candidates.truncate(limit.max(1));
    candidates
}

fn frequency_pressure_swaps(
    alphabet: &[u8; 26],
    values: &[usize],
    shifts: &[usize],
    limit: usize,
) -> Vec<[u8; 26]> {
    let plaintext = decode_keyed(values, shifts, alphabet);
    if plaintext.len() < 20 {
        return Vec::new();
    }
    let mut counts = [0usize; 26];
    for ch in plaintext.bytes() {
        if ch.is_ascii_uppercase() {
            counts[(ch - b'A') as usize] += 1;
        }
    }
    let n = plaintext.len() as f64;
    let mut over = Vec::new();
    let mut under = Vec::new();
    for i in 0..26 {
        let expected = ENGLISH_FREQS[i] * n;
        if expected <= 0.0 {
            continue;
        }
        let delta = (counts[i] as f64 - expected) / expected.sqrt();
        if delta > 0.0 {
            over.push((delta, b'A' + i as u8));
        } else {
            under.push((-delta, b'A' + i as u8));
        }
    }
    over.sort_by(|a, b| b.0.total_cmp(&a.0));
    under.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut proposals = Vec::new();
    for &(_, a) in over.iter().take(5) {
        for &(_, b) in under.iter().take(5) {
            let proposal = swap_letters(*alphabet, a, b);
            if &proposal != alphabet {
                proposals.push(proposal);
                if proposals.len() >= limit {
                    return proposals;
                }
            }
        }
    }
    proposals
}

fn phase_common_letter_swaps(
    alphabet: &[u8; 26],
    values: &[usize],
    period: usize,
    shifts: &[usize],
    limit: usize,
) -> Vec<[u8; 26]> {
    let mut proposals = Vec::new();
    for phase in 0..period.max(1) {
        let mut counts = [0usize; 26];
        for c in values.iter().skip(phase).step_by(period) {
            counts[*c] += 1;
        }
        let mut top: Vec<(usize, usize)> =
            counts.iter().enumerate().map(|(i, c)| (i, *c)).collect();
        top.sort_by(|a, b| b.1.cmp(&a.1));
        let shift = shifts[phase % shifts.len()];
        for &(cipher_idx, count) in top.iter().take(4) {
            if count == 0 {
                continue;
            }
            let current = alphabet[(cipher_idx + 26 - shift) % 26];
            for target in ENGLISH_FREQ_ORDER.iter().take(8) {
                if current == *target {
                    continue;
                }
                let proposal = swap_letters(*alphabet, current, *target);
                if &proposal != alphabet {
                    proposals.push(proposal);
                    if proposals.len() >= limit {
                        return proposals;
                    }
                }
            }
        }
    }
    proposals
}

fn swap_letters(mut alphabet: [u8; 26], a: u8, b: u8) -> [u8; 26] {
    if a == b {
        return alphabet;
    }
    let i = alphabet.iter().position(|ch| *ch == a);
    let j = alphabet.iter().position(|ch| *ch == b);
    if let (Some(i), Some(j)) = (i, j) {
        alphabet.swap(i, j);
    }
    alphabet
}

fn mutate_alphabet(mut alphabet: [u8; 26], rng: &mut StdRng) -> [u8; 26] {
    let roll = rng.gen::<f64>();
    if roll < 0.70 {
        let i = rng.gen_range(0..26);
        let mut j = rng.gen_range(0..26);
        while i == j {
            j = rng.gen_range(0..26);
        }
        alphabet.swap(i, j);
    } else if roll < 0.90 {
        let i = rng.gen_range(0..26);
        let mut j = rng.gen_range(0..26);
        while i == j {
            j = rng.gen_range(0..26);
        }
        let ch = alphabet[i];
        if i < j {
            for k in i..j {
                alphabet[k] = alphabet[k + 1];
            }
        } else {
            for k in (j + 1..=i).rev() {
                alphabet[k] = alphabet[k - 1];
            }
        }
        alphabet[j] = ch;
    } else {
        let i = rng.gen_range(0..26);
        let j = rng.gen_range(0..26);
        let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
        alphabet[lo..=hi].reverse();
    }
    alphabet
}

fn scramble_alphabet(mut alphabet: [u8; 26], rng: &mut StdRng, swaps: usize) -> [u8; 26] {
    for _ in 0..swaps {
        let i = rng.gen_range(0..26);
        let mut j = rng.gen_range(0..26);
        while i == j {
            j = rng.gen_range(0..26);
        }
        alphabet.swap(i, j);
    }
    alphabet
}

fn key_string(shifts: &[usize], alphabet: &[u8; 26]) -> String {
    shifts.iter().map(|s| alphabet[s % 26] as char).collect()
}

fn period_complexity_penalty(period: usize) -> f64 {
    (period as f64).ln() * 0.02
}

fn round5(value: f64) -> f64 {
    (value * 100000.0).round() / 100000.0
}

#[pymodule]
fn decipher_fast(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(normalized_ngram_score, m)?)?;
    m.add_function(wrap_pyfunction!(keyed_vigenere_alphabet_anneal, m)?)?;
    m.add_function(wrap_pyfunction!(quagmire3_shotgun_search, m)?)?;
    m.add_function(wrap_pyfunction!(transform_structural_metrics_batch, m)?)?;
    Ok(())
}
