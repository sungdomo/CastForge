//! Length-prefixed binary trace recording, tagged argument codec, and GPU
//! transfer phase timing sampling.
//!
//! Encodes/decodes primitive values, tuples, and typed domain objects via
//! `{"__t__": tag, "v": payload}` envelopes, lazily writes the 4-byte
//! big-endian length-prefixed header with SHA-256 config digest on the first
//! of `attach_storage_config`, `write_record`, or `close()`, and pops GPU
//! gather/DMA phase timing samples on `MP_STORE_END` / `MP_RETRIEVE_END`
//! (emitting `MP_TRANSFER_PHASE_SAMPLES` even when the sample batch is empty).

use std::collections::{BTreeMap, VecDeque};

use sha2::{Digest, Sha256};

/// Magic header prefix for binary trace files.
pub const TRACE_FILE_MAGIC: &[u8; 4] = b"CFTR";

/// Trace value representation supporting native primitives, tuples, dicts, and
/// tagged custom domain codecs (`{"__t__": tag, "v": payload}`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceValue {
    /// Null/None value.
    Null,
    /// Boolean scalar.
    Bool(bool),
    /// Signed 64-bit integer.
    Int(i64),
    /// UTF-8 string.
    Str(String),
    /// Ordered list of trace values.
    List(Vec<TraceValue>),
    /// Ordered tuple preserved across serialization via `{"__t__": "tuple", "v": [...]}`.
    Tuple(Vec<TraceValue>),
    /// String-keyed dictionary.
    Dict(BTreeMap<String, TraceValue>),
    /// Typed domain object encoded as `{"__t__": tag, "v": fields}`.
    TaggedObject {
        /// Registered type tag (e.g., `"ObjectKey"`, `"MemoryLayoutDesc"`,
        /// `"PrefetchHandle"`, `"AttnWindowDesc"`).
        tag: String,
        /// Encoded payload fields.
        payload: BTreeMap<String, TraceValue>,
    },
}

/// Encodes a [`TraceValue`] into its canonical wire representation where
/// [`TraceValue::Tuple`] and [`TraceValue::TaggedObject`] are wrapped in
/// `{"__t__": tag, "v": ...}` dictionaries.
#[must_use]
pub fn encode_trace_value(val: &TraceValue) -> TraceValue {
    match val {
        TraceValue::Null | TraceValue::Bool(_) | TraceValue::Int(_) | TraceValue::Str(_) => {
            val.clone()
        }
        TraceValue::List(items) => TraceValue::List(items.iter().map(encode_trace_value).collect()),
        TraceValue::Tuple(items) => {
            let mut wrapper = BTreeMap::new();
            wrapper.insert("__t__".to_string(), TraceValue::Str("tuple".to_string()));
            wrapper.insert(
                "v".to_string(),
                TraceValue::List(items.iter().map(encode_trace_value).collect()),
            );
            TraceValue::Dict(wrapper)
        }
        TraceValue::Dict(map) => {
            let mut out = BTreeMap::new();
            for (k, v) in map {
                out.insert(k.clone(), encode_trace_value(v));
            }
            TraceValue::Dict(out)
        }
        TraceValue::TaggedObject { tag, payload } => {
            let mut inner = BTreeMap::new();
            for (k, v) in payload {
                inner.insert(k.clone(), encode_trace_value(v));
            }
            let mut wrapper = BTreeMap::new();
            wrapper.insert("__t__".to_string(), TraceValue::Str(tag.clone()));
            wrapper.insert("v".to_string(), TraceValue::Dict(inner));
            TraceValue::Dict(wrapper)
        }
    }
}

/// Decodes a wire-encoded [`TraceValue`] back into its native [`TraceValue`]
/// form, recognizing `{"__t__": "tuple", "v": [...]}` and registered domain
/// tags (`"ObjectKey"`, `"MemoryLayoutDesc"`, `"PrefetchHandle"`,
/// `"AttnWindowDesc"`).
///
/// # Errors
///
/// Returns `Err` if a wrapped dictionary carries an unknown `__t__` tag.
pub fn decode_trace_value(val: &TraceValue) -> Result<TraceValue, String> {
    match val {
        TraceValue::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(decode_trace_value(item)?);
            }
            Ok(TraceValue::List(out))
        }
        TraceValue::Dict(map) => {
            if let Some(TraceValue::Str(tag)) = map.get("__t__") {
                let Some(raw_v) = map.get("v") else {
                    return Err(format!("trace.codecs: missing 'v' field for tag '{tag}'"));
                };
                if tag == "tuple" {
                    let TraceValue::List(items) = raw_v else {
                        return Err("trace.codecs: tuple payload must be a list".to_string());
                    };
                    let mut decoded = Vec::with_capacity(items.len());
                    for item in items {
                        decoded.push(decode_trace_value(item)?);
                    }
                    return Ok(TraceValue::Tuple(decoded));
                }
                let is_known_tag = matches!(
                    tag.as_str(),
                    "ObjectKey" | "MemoryLayoutDesc" | "PrefetchHandle" | "AttnWindowDesc"
                );
                if !is_known_tag {
                    return Err(format!("trace.codecs: unknown tag '{tag}'"));
                }
                let TraceValue::Dict(inner_map) = raw_v else {
                    return Err(format!(
                        "trace.codecs: payload for tag '{tag}' must be a dict"
                    ));
                };
                let mut decoded_inner = BTreeMap::new();
                for (k, v) in inner_map {
                    decoded_inner.insert(k.clone(), decode_trace_value(v)?);
                }
                return Ok(TraceValue::TaggedObject {
                    tag: tag.clone(),
                    payload: decoded_inner,
                });
            }
            let mut out = BTreeMap::new();
            for (k, v) in map {
                out.insert(k.clone(), decode_trace_value(v)?);
            }
            Ok(TraceValue::Dict(out))
        }
        other => Ok(other.clone()),
    }
}

/// Binary trace recorder that lazily writes a 4-byte big-endian length-prefixed
/// header (with SHA-256 config digest) on the first of `attach_storage_config`,
/// `write_record`, or `close()`, and frames every call record with relative
/// monotonic timestamps.
#[derive(Debug, Clone)]
pub struct BinaryTraceRecorder {
    level: String,
    captures_calls: bool,
    tracing_gate_enabled: bool,
    header_written: bool,
    closed: bool,
    t_mono_start_ns: u64,
    sm_config_json: String,
    sm_config_digest: String,
    dropped_count: usize,
    stream_bytes: Vec<u8>,
}

impl BinaryTraceRecorder {
    /// Creates a new binary trace recorder starting at `t_mono_start_ns`.
    /// When `captures_calls` is `true`, flips `tracing_gate_enabled` on.
    #[must_use]
    pub fn new(level: &str, captures_calls: bool, t_mono_start_ns: u64) -> Self {
        Self {
            level: level.to_string(),
            captures_calls,
            tracing_gate_enabled: captures_calls,
            header_written: false,
            closed: false,
            t_mono_start_ns,
            sm_config_json: String::new(),
            sm_config_digest: String::new(),
            dropped_count: 0,
            stream_bytes: Vec::new(),
        }
    }

    /// Returns whether the global call-tracing gate is currently enabled.
    #[must_use]
    pub const fn tracing_gate_enabled(&self) -> bool {
        self.tracing_gate_enabled
    }

    /// Returns whether the header frame has been written.
    #[must_use]
    pub const fn header_written(&self) -> bool {
        self.header_written
    }

    /// Returns the SHA-256 hex digest of the attached storage configuration.
    #[must_use]
    pub fn sm_config_digest(&self) -> &str {
        &self.sm_config_digest
    }

    /// Returns the number of dropped records.
    #[must_use]
    pub const fn dropped_count(&self) -> usize {
        self.dropped_count
    }

    /// Attaches a sorted-key storage configuration map, computing its SHA-256
    /// digest and writing the header frame once. Subsequent calls are ignored.
    pub fn attach_storage_config(&mut self, config_kv: &BTreeMap<String, String>) {
        if self.closed || self.header_written {
            return;
        }
        let mut json = String::from("{");
        for (idx, (k, v)) in config_kv.iter().enumerate() {
            if idx > 0 {
                json.push(',');
            }
            json.push_str(&format!("\"{k}\":\"{v}\""));
        }
        json.push('}');

        let mut hasher = Sha256::new();
        hasher.update(json.as_bytes());
        let digest_bytes = hasher.finalize();
        let mut hex = String::with_capacity(64);
        for b in digest_bytes {
            use std::fmt::Write;
            let _ = write!(&mut hex, "{b:02x}");
        }
        self.sm_config_json = json;
        self.sm_config_digest = hex;
        self.emit_header_frame();
    }

    fn emit_header_frame(&mut self) {
        if self.header_written {
            return;
        }
        let payload = format!(
            "HDR|magic=CFTR|level={}|digest={}|cfg={}",
            self.level, self.sm_config_digest, self.sm_config_json
        );
        self.write_len_prefixed_frame(payload.as_bytes());
        self.header_written = true;
    }

    fn write_len_prefixed_frame(&mut self, frame: &[u8]) {
        let len_be = (frame.len() as u32).to_be_bytes();
        self.stream_bytes.extend_from_slice(&len_be);
        self.stream_bytes.extend_from_slice(frame);
    }

    /// Appends a `TRACE_CALL` record with monotonic timestamp `t_mono_ns`
    /// stored relative to `t_mono_start_ns`. If no header has been written yet,
    /// writes the fallback header first.
    pub fn write_record(
        &mut self,
        qualname: &str,
        args: &BTreeMap<String, TraceValue>,
        t_mono_ns: u64,
    ) {
        if self.closed {
            return;
        }
        if qualname.trim().is_empty() {
            self.dropped_count += 1;
            return;
        }
        if !self.header_written {
            self.emit_header_frame();
        }
        let rel_mono_ns = t_mono_ns.saturating_sub(self.t_mono_start_ns);
        let mut encoded_args = BTreeMap::new();
        for (k, v) in args {
            encoded_args.insert(k.clone(), encode_trace_value(v));
        }
        let frame_str = format!("REC|t_rel={rel_mono_ns}|fn={qualname}|args={encoded_args:?}");
        self.write_len_prefixed_frame(frame_str.as_bytes());
    }

    /// Closes the trace recorder idempotently, flipping the tracing gate off
    /// and writing a fallback empty-config header if neither
    /// `attach_storage_config` nor `write_record` was called.
    pub fn close(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        if self.captures_calls {
            self.tracing_gate_enabled = false;
        }
        if !self.header_written {
            self.emit_header_frame();
        }
    }

    /// Parses the recorded byte stream into individual frame strings by reading
    /// each 4-byte big-endian length prefix, tolerating a truncated tail frame.
    #[must_use]
    pub fn read_frames(&self) -> Vec<String> {
        let mut frames = Vec::new();
        let mut offset = 0usize;
        while offset + 4 <= self.stream_bytes.len() {
            let mut len_bytes = [0u8; 4];
            len_bytes.copy_from_slice(&self.stream_bytes[offset..offset + 4]);
            let frame_len = u32::from_be_bytes(len_bytes) as usize;
            offset += 4;
            if offset + frame_len > self.stream_bytes.len() {
                // Gracefully tolerate partial-write tail truncation.
                break;
            }
            if let Ok(s) = std::str::from_utf8(&self.stream_bytes[offset..offset + frame_len]) {
                frames.push(s.to_string());
            }
            offset += frame_len;
        }
        frames
    }
}

/// GPU gather/DMA transfer phase timing sample.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferPhaseSample {
    /// Phase name (e.g., `"gpu_gather"`, `"host_dma"`).
    pub phase_name: String,
    /// Duration in microseconds.
    pub duration_us: u64,
}

/// Published `MP_TRANSFER_PHASE_SAMPLES` event emitted by [`TransferPhaseTimingSampler`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferPhaseSamplesEvent {
    /// Transfer key that just finished (`ended_transfer_key`).
    pub ended_transfer_key: String,
    /// Drained phase timing samples (may be empty, which still signals transfer completion).
    pub samples: Vec<TransferPhaseSample>,
}

/// Bridges the native GPU transfer phase-timing queue to the observability
/// event bus on `MP_STORE_END` and `MP_RETRIEVE_END`.
#[derive(Debug, Default)]
pub struct TransferPhaseTimingSampler {
    native_timing_enabled: bool,
    completed_queue: VecDeque<TransferPhaseSample>,
    published_events: Vec<TransferPhaseSamplesEvent>,
}

impl TransferPhaseTimingSampler {
    /// Creates a new phase-timing sampler (`native_timing_enabled` controls
    /// whether `pop_completed_phase_timings` is active).
    #[must_use]
    pub fn new(native_timing_enabled: bool) -> Self {
        Self {
            native_timing_enabled,
            completed_queue: VecDeque::new(),
            published_events: Vec::new(),
        }
    }

    /// Pushes a completed GPU phase timing sample into the native recorder queue.
    pub fn record_native_sample(&mut self, phase_name: &str, duration_us: u64) {
        self.completed_queue.push_back(TransferPhaseSample {
            phase_name: phase_name.to_string(),
            duration_us,
        });
    }

    /// Handles an `MP_STORE_END` or `MP_RETRIEVE_END` event for `transfer_key`,
    /// popping all finished samples and publishing an `MP_TRANSFER_PHASE_SAMPLES`
    /// event even when the popped sample list is empty.
    pub fn on_transfer_end(&mut self, transfer_key: &str) {
        if !self.native_timing_enabled {
            return;
        }
        let samples: Vec<TransferPhaseSample> = self.completed_queue.drain(..).collect();
        self.published_events.push(TransferPhaseSamplesEvent {
            ended_transfer_key: transfer_key.to_string(),
            samples,
        });
    }

    /// Returns all published `MP_TRANSFER_PHASE_SAMPLES` events.
    #[must_use]
    pub fn published_events(&self) -> &[TransferPhaseSamplesEvent] {
        &self.published_events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_codec_and_binary_recorder() {
        let val = TraceValue::Tuple(vec![
            TraceValue::Int(42),
            TraceValue::TaggedObject {
                tag: "ObjectKey".to_string(),
                payload: BTreeMap::from([(
                    "model_name".to_string(),
                    TraceValue::Str("qwen".to_string()),
                )]),
            },
        ]);
        let wire = encode_trace_value(&val);
        assert_eq!(decode_trace_value(&wire), Ok(val));

        let mut rec = BinaryTraceRecorder::new("storage", true, 1_000);
        assert!(rec.tracing_gate_enabled());
        assert!(!rec.header_written());
        let mut cfg = BTreeMap::new();
        cfg.insert("l1_gb".to_string(), "64".to_string());
        rec.attach_storage_config(&cfg);
        assert!(rec.header_written());
        rec.write_record("storage.put", &BTreeMap::new(), 1_500);
        rec.close();
        assert!(!rec.tracing_gate_enabled());
        assert_eq!(rec.read_frames().len(), 2);
    }
}
