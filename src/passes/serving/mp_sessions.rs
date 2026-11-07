//! Multiprocess request sessions, rolling token chunk hashing, transport API-key auth, and IPC
//! policy.
//!
//! Implements two multiprocess KV-serving subsystems:
//! 1. **Per-Request Token & Rolling-Hash Session Manager (`MpRequestSession`,
//!    `MpSessionManager`)**:
//!    - Holds packed 4-byte token sequences (`MP_TOKEN_STRIDE_BYTES = 4`) and computes incremental
//!      rolling chunk hashes (`get_hashes`) that reuse previously hashed prefix chunks.
//!    - Tracks per-lookup prefetch hit counts, locked GIDs, and retained L1 read-lock owners
//!      (`begin_lookup`, `record_prefetch_result`, `get_prefetch_owners`), and expires idle
//!      sessions by TTL (`cleanup_expired`) while invoking registered destroy callbacks.
//! 2. **Transport API-Key Rotation Verifier & Runtime IPC Policy (`ApiKeyVerifier`, `IpcPolicy`)**:
//!    - Loads multi-key service rotation credentials from CLI files and environment sources,
//!      performs constant-time token verification (`verify_token`) with secret-redacted `Debug`
//!      and `Display` formatting, builds outbound `Authorization: Bearer` headers, and manages
//!      container-isolation (`isolated_ipc`) and VMM (`use_vmm_api`) IPC policy flags.

use indexmap::IndexMap;
use std::fmt;

/// Byte width of a single packed token ID (`4` bytes / `u32`).
pub const MP_TOKEN_STRIDE_BYTES: usize = 4;

/// Default time-to-live in seconds for a multiprocess request session (`600.0` seconds).
pub const DEFAULT_MP_SESSION_TTL_SECS: f64 = 600.0;

/// Mixes a packed token chunk (`chunk_bytes`) with `prefix_hash` using a 64-bit FNV-1a / SplitMix64
/// rolling combiner.
#[must_use]
pub fn hash_packed_token_chunk_with_prefix(chunk_bytes: &[u8], prefix_hash: u64) -> u64 {
    let mut h = prefix_hash ^ 0xCBF2_9CE4_8422_2325;
    for &b in chunk_bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01B3);
    }
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^ (h >> 31)
}

/// Per-request session state tracking packed token IDs, incremental rolling chunk hashes, and
/// active prefetch L1 read-lock ownership.
#[derive(Debug, Clone)]
pub struct MpRequestSession {
    /// Unique request identifier.
    pub request_id: String,
    /// Number of tokens per KV chunk.
    pub chunk_size: usize,
    /// Initial seed hash used for the first chunk in a sequence.
    pub none_hash: u64,
    /// Packed token IDs (`MP_TOKEN_STRIDE_BYTES` bytes per token).
    pub tokens: Vec<u8>,
    /// Cached rolling hashes for chunks `0..num_chunks_processed`.
    pub chunk_hashes: Vec<u64>,
    /// Rolling prefix hash from the most recently hashed chunk.
    pub last_prefix_hash: Option<u64>,
    /// Number of full chunks already hashed into `chunk_hashes`.
    pub num_chunks_processed: usize,
    /// Timestamp (in seconds) when this session was created.
    pub created_at_secs: f64,
    /// Optional IPC cache server lookup key for the in-flight lookup.
    pub lookup_ipc_key: Option<String>,
    /// Number of prefetch hit chunks recorded by the current lookup (`-1` if not yet recorded).
    pub prefetch_hit_chunks: i64,
    /// Locked GPU/buffer group IDs retained by the current lookup.
    pub prefetch_locked_gids: Vec<i64>,
    prefetch_owners: IndexMap<String, usize>,
    /// Per-request extension metadata owned by storage/lookup subsystems.
    pub extras: IndexMap<String, String>,
}

impl MpRequestSession {
    /// Creates a new [`MpRequestSession`] for `request_id` with `chunk_size` tokens per chunk.
    pub fn new(
        request_id: impl Into<String>,
        chunk_size: usize,
        none_hash: u64,
        created_at_secs: f64,
    ) -> Result<Self, String> {
        if chunk_size == 0 {
            return Err("chunk_size must be positive".to_string());
        }
        Ok(Self {
            request_id: request_id.into(),
            chunk_size,
            none_hash,
            tokens: Vec::new(),
            chunk_hashes: Vec::new(),
            last_prefix_hash: None,
            num_chunks_processed: 0,
            created_at_secs,
            lookup_ipc_key: None,
            prefetch_hit_chunks: -1,
            prefetch_locked_gids: Vec::new(),
            prefetch_owners: IndexMap::new(),
            extras: IndexMap::new(),
        })
    }

    /// Replaces the session's packed token sequence (idempotent replacement starting at token 0).
    pub fn set_tokens(&mut self, packed_token_ids: Vec<u8>) -> Result<(), String> {
        if !packed_token_ids.len().is_multiple_of(MP_TOKEN_STRIDE_BYTES) {
            return Err(format!(
                "packed_token_ids length ({}) must be a multiple of {}",
                packed_token_ids.len(),
                MP_TOKEN_STRIDE_BYTES
            ));
        }
        self.tokens = packed_token_ids;
        Ok(())
    }

    /// Returns the number of packed tokens currently held by this session.
    #[must_use]
    pub fn num_held_tokens(&self) -> usize {
        self.tokens.len() / MP_TOKEN_STRIDE_BYTES
    }

    /// Computes and returns rolling chunk hashes for the token range `[start, end)`:
    /// - Both `start` and `end` must be multiples of `self.chunk_size`.
    /// - When `end` is `None`, it defaults to the last full-chunk boundary of `self.tokens`.
    /// - Returns an error if an explicit `end` exceeds `self.num_held_tokens()`.
    /// - Skips already-computed chunks (`0..self.num_chunks_processed`).
    pub fn get_hashes(&mut self, start: usize, end: Option<usize>) -> Result<Vec<u64>, String> {
        if !start.is_multiple_of(self.chunk_size) {
            return Err(format!(
                "start ({start}) must be a multiple of chunk_size ({})",
                self.chunk_size
            ));
        }
        let held = self.num_held_tokens();
        let resolved_end = match end {
            Some(explicit_end) => {
                if explicit_end > held {
                    return Err(format!(
                        "get_hashes end ({explicit_end}) exceeds the session's {held} token(s); \
                         the session may have been recreated after request cleanup"
                    ));
                }
                if !explicit_end.is_multiple_of(self.chunk_size) {
                    return Err(format!(
                        "end ({explicit_end}) must be a multiple of chunk_size ({})",
                        self.chunk_size
                    ));
                }
                explicit_end
            }
            None => held - (held % self.chunk_size),
        };
        if resolved_end < start {
            return Err(format!(
                "resolved end ({resolved_end}) is less than start ({start})"
            ));
        }

        let start_chunk = start / self.chunk_size;
        let end_chunk = resolved_end / self.chunk_size;
        self.compute_hash_up_to(end_chunk);
        Ok(self.chunk_hashes[start_chunk..end_chunk].to_vec())
    }

    fn compute_hash_up_to(&mut self, end_chunk: usize) {
        let stride_bytes = self.chunk_size * MP_TOKEN_STRIDE_BYTES;
        while self.num_chunks_processed < end_chunk {
            let byte_start = self.num_chunks_processed * stride_bytes;
            let chunk_slice = &self.tokens[byte_start..byte_start + stride_bytes];
            let prefix = self.last_prefix_hash.unwrap_or(self.none_hash);
            let h = hash_packed_token_chunk_with_prefix(chunk_slice, prefix);
            self.last_prefix_hash = Some(h);
            self.chunk_hashes.push(h);
            self.num_chunks_processed += 1;
        }
    }

    /// Records the start of a new lookup and resets the previous lookup's prefetch state.
    pub fn begin_lookup(&mut self, ipc_key: impl Into<String>) {
        self.lookup_ipc_key = Some(ipc_key.into());
        self.prefetch_hit_chunks = -1;
        self.prefetch_locked_gids.clear();
        self.prefetch_owners.clear();
    }

    /// Records the lock set and L1 owner counts acquired by the current lookup.
    pub fn record_prefetch_result(
        &mut self,
        hit_chunks: i64,
        locked_gids: Vec<i64>,
        l1_owners: Option<IndexMap<String, usize>>,
    ) {
        self.prefetch_hit_chunks = hit_chunks;
        self.prefetch_locked_gids = locked_gids;
        self.prefetch_owners = l1_owners.unwrap_or_default();
    }

    /// Returns a snapshot of the L1 object keys and owner counts retained by this session's lookup.
    #[must_use]
    pub fn get_prefetch_owners(&self) -> IndexMap<String, usize> {
        self.prefetch_owners.clone()
    }
}

type SessionDestroyListener = Box<dyn FnMut(&MpRequestSession) + Send>;

/// Manager for per-request [`MpRequestSession`] instances with TTL expiration and destroy
/// notifications.
pub struct MpSessionManager {
    chunk_size: usize,
    none_hash: u64,
    ttl_secs: f64,
    sessions: IndexMap<String, MpRequestSession>,
    destroy_listeners: Vec<SessionDestroyListener>,
    destroyed_request_ids: Vec<String>,
}

impl fmt::Debug for MpSessionManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MpSessionManager")
            .field("chunk_size", &self.chunk_size)
            .field("ttl_secs", &self.ttl_secs)
            .field("active_sessions", &self.sessions.len())
            .field("destroy_listener_count", &self.destroy_listeners.len())
            .finish()
    }
}

impl MpSessionManager {
    /// Creates a new [`MpSessionManager`] with `chunk_size` tokens per chunk and session TTL
    /// `ttl_secs`.
    pub fn new(chunk_size: usize, none_hash: u64, ttl_secs: f64) -> Result<Self, String> {
        if chunk_size == 0 {
            return Err("chunk_size must be positive".to_string());
        }
        Ok(Self {
            chunk_size,
            none_hash,
            ttl_secs,
            sessions: IndexMap::new(),
            destroy_listeners: Vec::new(),
            destroyed_request_ids: Vec::new(),
        })
    }

    /// Registers a callback invoked whenever a session is removed or expires.
    pub fn add_destroy_listener<F>(&mut self, listener: F)
    where
        F: FnMut(&MpRequestSession) + Send + 'static,
    {
        self.destroy_listeners.push(Box::new(listener));
    }

    /// Gets an existing session for `request_id` or creates a new one stamped with `now_secs`.
    pub fn get_or_create(&mut self, request_id: &str, now_secs: f64) -> &mut MpRequestSession {
        if !self.sessions.contains_key(request_id) {
            let session = MpRequestSession::new(
                request_id.to_string(),
                self.chunk_size,
                self.none_hash,
                now_secs,
            )
            .expect("chunk_size validated at MpSessionManager construction");
            self.sessions.insert(request_id.to_string(), session);
        }
        self.sessions
            .get_mut(request_id)
            .expect("session just inserted")
    }

    /// Returns an immutable reference to an existing session without creating one.
    #[must_use]
    pub fn get(&self, request_id: &str) -> Option<&MpRequestSession> {
        self.sessions.get(request_id)
    }

    /// Returns a mutable reference to an existing session without creating one.
    pub fn get_mut(&mut self, request_id: &str) -> Option<&mut MpRequestSession> {
        self.sessions.get_mut(request_id)
    }

    /// Removes a session by `request_id` and notifies all registered destroy listeners.
    pub fn remove(&mut self, request_id: &str) -> Option<MpRequestSession> {
        let removed = self.sessions.shift_remove(request_id)?;
        self.notify_destroyed(&removed);
        Some(removed)
    }

    /// Removes all sessions whose age (`now_secs - session.created_at_secs`) strictly exceeds
    /// `self.ttl_secs`, invoking all destroy listeners for each expired session.
    ///
    /// Returns the number of expired sessions removed.
    pub fn cleanup_expired(&mut self, now_secs: f64) -> usize {
        let expired_keys: Vec<String> = self
            .sessions
            .iter()
            .filter_map(|(k, s)| {
                if now_secs - s.created_at_secs > self.ttl_secs {
                    Some(k.clone())
                } else {
                    None
                }
            })
            .collect();

        let mut expired_sessions = Vec::with_capacity(expired_keys.len());
        for key in expired_keys {
            if let Some(s) = self.sessions.shift_remove(&key) {
                expired_sessions.push(s);
            }
        }

        for session in &expired_sessions {
            self.notify_destroyed(session);
        }
        expired_sessions.len()
    }

    fn notify_destroyed(&mut self, session: &MpRequestSession) {
        self.destroyed_request_ids.push(session.request_id.clone());
        for listener in &mut self.destroy_listeners {
            listener(session);
        }
    }

    /// Returns the number of currently active sessions.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.sessions.len()
    }

    /// Returns the history of request IDs whose sessions have been destroyed.
    #[must_use]
    pub fn destroyed_request_ids(&self) -> &[String] {
        &self.destroyed_request_ids
    }
}

/// Validates and extracts a single-line API key from file contents.
pub fn parse_single_line_api_key(contents: &str, source_label: &str) -> Result<String, String> {
    let trimmed = contents.trim();
    if trimmed.is_empty() {
        return Err(format!("API key file is empty: {source_label}"));
    }
    if trimmed.lines().count() != 1 {
        return Err(format!(
            "API key file must contain exactly one line: {source_label}"
        ));
    }
    Ok(trimmed.to_string())
}

/// Loads accepted service API keys in cumulative rotation order:
/// 1. Command-line key files (`cli_key_files`: `(path_label, file_contents)`).
/// 2. Optional environment key file (`env_key_file`: `(path_label, file_contents)`).
/// 3. Optional environment key value (`env_key_value`, ignoring empty strings).
pub fn load_service_api_keys(
    cli_key_files: &[(&str, &str)],
    env_key_file: Option<(&str, &str)>,
    env_key_value: Option<&str>,
) -> Result<Vec<String>, String> {
    let mut keys = Vec::with_capacity(cli_key_files.len() + 2);
    for &(label, content) in cli_key_files {
        keys.push(parse_single_line_api_key(content, label)?);
    }
    if let Some((label, content)) = env_key_file
        && !label.is_empty()
    {
        keys.push(parse_single_line_api_key(content, label)?);
    }
    if let Some(env_val) = env_key_value
        && !env_val.is_empty()
    {
        keys.push(env_val.to_string());
    }
    Ok(keys)
}

/// Loads the single API key sent by a request client:
/// - Explicit `explicit_api_key` takes precedence (returns an error if empty `""`).
/// - Otherwise uses the first configured CLI key file (`cli_key_files[0]`) while validating all
///   configured key files so server rotation files stay well-formed.
/// - Otherwise falls back to environment sources, returning an error if more than one environment
///   key source is configured simultaneously.
pub fn load_request_client_api_key(
    explicit_api_key: Option<&str>,
    cli_key_files: &[(&str, &str)],
    env_key_file: Option<(&str, &str)>,
    env_key_value: Option<&str>,
) -> Result<Option<String>, String> {
    if let Some(explicit) = explicit_api_key {
        if explicit.is_empty() {
            return Err("Request client API key must not be empty".to_string());
        }
        return Ok(Some(explicit.to_string()));
    }
    if !cli_key_files.is_empty() {
        let mut parsed = Vec::with_capacity(cli_key_files.len());
        for &(label, content) in cli_key_files {
            parsed.push(parse_single_line_api_key(content, label)?);
        }
        return Ok(parsed.into_iter().next());
    }
    let env_keys = load_service_api_keys(&[], env_key_file, env_key_value)?;
    if env_keys.len() > 1 {
        return Err("Configure exactly one request client API key source".to_string());
    }
    Ok(env_keys.into_iter().next())
}

/// Builds outbound HTTP authentication headers (`{"Authorization": "Bearer <key>"}`) for a
/// multiprocess request client, or an empty map when no API key is configured.
pub fn request_client_auth_headers(
    explicit_api_key: Option<&str>,
    cli_key_files: &[(&str, &str)],
    env_key_file: Option<(&str, &str)>,
    env_key_value: Option<&str>,
) -> Result<IndexMap<String, String>, String> {
    let mut headers = IndexMap::new();
    if let Some(key) =
        load_request_client_api_key(explicit_api_key, cli_key_files, env_key_file, env_key_value)?
    {
        headers.insert("Authorization".to_string(), format!("Bearer {key}"));
    }
    Ok(headers)
}

#[must_use]
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Verifies caller-supplied bearer tokens against one or more accepted rotation API keys in
/// constant time per key, never exposing key material in `Debug` or `Display` representations.
#[derive(Clone)]
pub struct ApiKeyVerifier {
    api_keys: Vec<Vec<u8>>,
}

impl ApiKeyVerifier {
    /// Creates a new [`ApiKeyVerifier`] from accepted plaintext API keys.
    ///
    /// Returns an error if any configured key is empty.
    pub fn new<I, S>(api_keys: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut encoded = Vec::new();
        for key in api_keys {
            let s = key.as_ref();
            if s.is_empty() {
                return Err("API keys must not be empty".to_string());
            }
            encoded.push(s.as_bytes().to_vec());
        }
        Ok(Self { api_keys: encoded })
    }

    /// Returns `true` when at least one API key is configured.
    #[must_use]
    pub fn enabled(&self) -> bool {
        !self.api_keys.is_empty()
    }

    /// Returns the number of accepted rotation keys.
    #[must_use]
    pub fn key_count(&self) -> usize {
        self.api_keys.len()
    }

    /// Verifies `token` in constant time against every configured API key without short-circuiting.
    #[must_use]
    pub fn verify_token(&self, token: Option<&str>) -> bool {
        let Some(tok_str) = token else {
            return false;
        };
        let tok_bytes = tok_str.as_bytes();
        let mut matched = false;
        for expected in &self.api_keys {
            matched |= constant_time_eq(tok_bytes, expected);
        }
        matched
    }
}

impl fmt::Debug for ApiKeyVerifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ApiKeyVerifier(enabled={}, key_count={})",
            self.enabled(),
            self.key_count()
        )
    }
}

impl fmt::Display for ApiKeyVerifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ApiKeyVerifier(enabled={}, key_count={})",
            self.enabled(),
            self.key_count()
        )
    }
}

/// Runtime IPC policy switches controlling cross-container isolation and VMM registration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IpcPolicy {
    /// Whether IPC handles must work across isolated container namespaces.
    pub isolated_ipc: bool,
    /// Whether KV buffer registration must use the VMM API instead of legacy IPC handles.
    pub use_vmm_api: bool,
}

impl IpcPolicy {
    /// Creates a default [`IpcPolicy`] (`isolated_ipc = false`, `use_vmm_api = false`).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            isolated_ipc: false,
            use_vmm_api: false,
        }
    }

    /// Updates optional policy fields in place and returns `self`.
    pub fn set(&mut self, isolated_ipc: Option<bool>, use_vmm_api: Option<bool>) -> Self {
        if let Some(iso) = isolated_ipc {
            self.isolated_ipc = iso;
        }
        if let Some(vmm) = use_vmm_api {
            self.use_vmm_api = vmm;
        }
        *self
    }

    /// Returns a copy with `isolated_ipc` set to `enabled`.
    #[must_use]
    pub const fn with_isolated_ipc(mut self, enabled: bool) -> Self {
        self.isolated_ipc = enabled;
        self
    }

    /// Returns a copy with `use_vmm_api` set to `enabled`.
    #[must_use]
    pub const fn with_use_vmm_api(mut self, enabled: bool) -> Self {
        self.use_vmm_api = enabled;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_mp_session_rolling_hashes_and_ttl_cleanup() {
        let mut mgr = MpSessionManager::new(2, 0x1234_5678, 60.0).expect("manager");
        let destroyed_log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let log_clone = Arc::clone(&destroyed_log);
        mgr.add_destroy_listener(move |sess| {
            log_clone
                .lock()
                .expect("lock")
                .push(sess.request_id.clone());
        });

        let sess = mgr.get_or_create("req-1", 100.0);
        let packed: Vec<u8> = (0..5u32).flat_map(u32::to_be_bytes).collect();
        sess.set_tokens(packed).expect("set_tokens");
        assert_eq!(sess.num_held_tokens(), 5);

        let hashes_all = sess.get_hashes(0, None).expect("hashes");
        assert_eq!(hashes_all.len(), 2);
        assert_eq!(sess.num_chunks_processed, 2);

        let second_only = sess.get_hashes(2, Some(4)).expect("second chunk");
        assert_eq!(second_only, vec![hashes_all[1]]);
        assert!(sess.get_hashes(0, Some(6)).is_err());

        let removed = mgr.cleanup_expired(161.0);
        assert_eq!(removed, 1);
        assert_eq!(mgr.active_count(), 0);
        assert_eq!(*destroyed_log.lock().expect("lock"), vec!["req-1"]);
    }

    #[test]
    fn test_api_key_verifier_rotation_and_redacted_debug() {
        let keys = load_service_api_keys(
            &[
                ("old.key", "secret-old-key\n"),
                ("new.key", "secret-new-key"),
            ],
            None,
            Some("secret-env-key"),
        )
        .expect("keys");
        assert_eq!(keys.len(), 3);

        let verifier = ApiKeyVerifier::new(&keys).expect("verifier");
        assert!(verifier.enabled());
        assert!(verifier.verify_token(Some("secret-old-key")));
        assert!(verifier.verify_token(Some("secret-new-key")));
        assert!(verifier.verify_token(Some("secret-env-key")));
        assert!(!verifier.verify_token(Some("wrong-key")));
        assert!(!verifier.verify_token(None));

        let dbg = format!("{verifier:?}");
        assert_eq!(dbg, "ApiKeyVerifier(enabled=true, key_count=3)");
        assert!(!dbg.contains("secret"));

        let headers = request_client_auth_headers(
            None,
            &[("old.key", "secret-old-key"), ("new.key", "secret-new-key")],
            None,
            None,
        )
        .expect("headers");
        assert_eq!(
            headers.get("Authorization").map(String::as_str),
            Some("Bearer secret-old-key")
        );
    }
}
