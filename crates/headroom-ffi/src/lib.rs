//! C FFI bindings for headroom-core compression.

use headroom_core::cache_control::compute_frozen_count;
use headroom_core::ccr::backends::{RedisCcrStore, SqliteCcrStore};
use headroom_core::ccr::{CcrStore, InMemoryCcrStore};
use headroom_core::tokenizer::{EstimatingCounter, Tokenizer};
use headroom_core::transforms::live_zone::{
    compress_anthropic_live_zone_with_ccr, compress_openai_chat_live_zone,
    compress_openai_responses_live_zone, AuthMode, LiveZoneOutcome,
};
use serde::Serialize;
use std::ffi::{c_char, CStr, CString};
use std::panic::catch_unwind;
use std::sync::{Arc, LazyLock, RwLock};

/// Default CCR backend: in-memory, lost on restart. Replaced via
/// `headroom_ccr_init_sqlite` for persistent storage.
static DEFAULT_CCR: LazyLock<Arc<dyn CcrStore>> =
    LazyLock::new(|| Arc::new(InMemoryCcrStore::default()));

/// Active CCR store. Initialized lazily to the default; swapped at runtime
/// when `headroom_ccr_init_sqlite` succeeds.
static CCR_STORE: LazyLock<RwLock<Arc<dyn CcrStore>>> =
    LazyLock::new(|| RwLock::new(DEFAULT_CCR.clone()));

fn current_ccr() -> Arc<dyn CcrStore> {
    CCR_STORE
        .read()
        .map(|guard| guard.clone())
        .unwrap_or_else(|_| DEFAULT_CCR.clone())
}

static TOKEN_COUNTER: LazyLock<EstimatingCounter> = LazyLock::new(EstimatingCounter::default);

/// Auth mode passed from Go (must match headroom-core enum order).
fn auth_from_u8(v: u8) -> AuthMode {
    match v {
        0 => AuthMode::Payg,
        1 => AuthMode::OAuth,
        2 => AuthMode::Subscription,
        _ => AuthMode::Unknown,
    }
}

#[derive(Serialize)]
struct CompressResult {
    modified: bool,
    body: Option<String>,
    tokens_before: usize,
    tokens_after: usize,
    tokens_saved: usize,
    compression_ratio: f64,
    error: Option<String>,
}

fn err_result(msg: impl Into<String>) -> CompressResult {
    CompressResult {
        modified: false,
        body: None,
        tokens_before: 0,
        tokens_after: 0,
        tokens_saved: 0,
        compression_ratio: 1.0,
        error: Some(msg.into()),
    }
}

fn err_result_with_tokens(msg: impl Into<String>, tokens: usize) -> CompressResult {
    CompressResult {
        modified: false,
        body: None,
        tokens_before: tokens,
        tokens_after: tokens,
        tokens_saved: 0,
        compression_ratio: 1.0,
        error: Some(msg.into()),
    }
}

fn count_tokens(text: &str) -> usize {
    TOKEN_COUNTER.count_text(text)
}

fn outcome_to_result(body_before: &str, outcome: LiveZoneOutcome) -> CompressResult {
    match outcome {
        LiveZoneOutcome::NoChange { .. } => {
            let tokens_before = count_tokens(body_before);
            CompressResult {
                modified: false,
                body: None,
                tokens_before,
                tokens_after: tokens_before,
                tokens_saved: 0,
                compression_ratio: 1.0,
                error: None,
            }
        }
        LiveZoneOutcome::Modified { new_body, .. } => {
            let tokens_before = count_tokens(body_before);
            let body_str = new_body.get();
            let tokens_after = count_tokens(body_str);
            let tokens_saved = tokens_before.saturating_sub(tokens_after);
            let ratio = if tokens_before > 0 {
                tokens_after as f64 / tokens_before as f64
            } else {
                1.0
            };
            CompressResult {
                modified: true,
                body: Some(body_str.to_string()),
                tokens_before,
                tokens_after,
                tokens_saved,
                compression_ratio: ratio,
                error: None,
            }
        }
    }
}

fn make_result(r: CompressResult) -> *mut c_char {
    let json = serde_json::to_string(&r).unwrap_or_else(|_| {
        r#"{"modified":false,"body":null,"tokens_before":0,"tokens_after":0,"tokens_saved":0,"compression_ratio":1.0,"error":"serde"}"#
            .to_string()
    });
    make_c_string(&json)
}

// ---------- C FFI ----------

/// # Safety
/// `body_ptr` must point to `body_len` valid UTF-8 bytes (or be NULL with
/// `body_len == 0`). `model` must be a valid null-terminated UTF-8 string.
/// The returned pointer must be freed with `headroom_result_free`.
#[no_mangle]
pub unsafe extern "C" fn headroom_compress_openai(
    body_ptr: *const u8,
    body_len: usize,
    model: *const c_char,
    auth_mode: u8,
) -> *mut c_char {
    let result: CompressResult = match catch_unwind(|| {
        let model_str = match ptr_to_string(model) {
            Ok(s) => s,
            Err(r) => return r,
        };
        let body = match ptr_to_string_bytes(body_ptr, body_len) {
            Ok(s) => s,
            Err(r) => return r,
        };
        let auth = auth_from_u8(auth_mode);
        match compress_openai_chat_live_zone(body.as_bytes(), auth, &model_str) {
            Ok(o) => outcome_to_result(&body, o),
            Err(e) => err_result_with_tokens(e.to_string(), count_tokens(&body)),
        }
    }) {
        Ok(r) => r,
        Err(_) => err_result("panic"),
    };
    make_result(result)
}

/// # Safety
/// Same constraints as `headroom_compress_openai`. `frozen_count` must not
/// exceed the number of messages in the body.
#[no_mangle]
pub unsafe extern "C" fn headroom_compress_anthropic(
    body_ptr: *const u8,
    body_len: usize,
    frozen_count: usize,
    model: *const c_char,
    auth_mode: u8,
) -> *mut c_char {
    let result: CompressResult = match catch_unwind(|| {
        let model_str = match ptr_to_string(model) {
            Ok(s) => s,
            Err(r) => return r,
        };
        let body = match ptr_to_string_bytes(body_ptr, body_len) {
            Ok(s) => s,
            Err(r) => return r,
        };
        // Honor any cache_control markers by raising the frozen floor —
        // this preserves the cached prefix so upstream prompt-cache hits
        // are not invalidated. Caller's frozen_count is also respected
        // (max of the two) for explicit few-shot pinning.
        let computed_floor = serde_json::from_slice::<serde_json::Value>(body.as_bytes())
            .ok()
            .map(|v| compute_frozen_count(&v))
            .unwrap_or(0);
        let effective_floor = frozen_count.max(computed_floor);
        let auth = auth_from_u8(auth_mode);
        let store = current_ccr();
        match compress_anthropic_live_zone_with_ccr(
            body.as_bytes(),
            effective_floor,
            auth,
            &model_str,
            Some(store.as_ref()),
        ) {
            Ok(o) => outcome_to_result(&body, o),
            Err(e) => err_result_with_tokens(e.to_string(), count_tokens(&body)),
        }
    }) {
        Ok(r) => r,
        Err(_) => err_result("panic"),
    };
    make_result(result)
}

#[derive(Serialize)]
struct CcrGetResult {
    found: bool,
    content: String,
    error: Option<String>,
}

/// # Safety
/// `hash` must be a valid null-terminated UTF-8 string. The returned pointer
/// must be freed with `headroom_result_free`.
#[no_mangle]
pub unsafe extern "C" fn headroom_ccr_get(hash: *const c_char) -> *mut c_char {
    let payload = match ptr_to_string(hash) {
        Ok(h) => match current_ccr().get(&h) {
            Some(content) => CcrGetResult {
                found: true,
                content,
                error: None,
            },
            None => CcrGetResult {
                found: false,
                content: String::new(),
                error: None,
            },
        },
        Err(r) => CcrGetResult {
            found: false,
            content: String::new(),
            error: r.error,
        },
    };
    let json = serde_json::to_string(&payload)
        .unwrap_or_else(|_| r#"{"found":false,"content":"","error":"serde"}"#.to_string());
    make_c_string(&json)
}

/// # Safety
/// Same constraints as `headroom_compress_openai`. The body must be a valid
/// OpenAI Responses-API payload (`input` array of items).
#[no_mangle]
pub unsafe extern "C" fn headroom_compress_openai_responses(
    body_ptr: *const u8,
    body_len: usize,
    model: *const c_char,
    auth_mode: u8,
) -> *mut c_char {
    let result: CompressResult = match catch_unwind(|| {
        let model_str = match ptr_to_string(model) {
            Ok(s) => s,
            Err(r) => return r,
        };
        let body = match ptr_to_string_bytes(body_ptr, body_len) {
            Ok(s) => s,
            Err(r) => return r,
        };
        let auth = auth_from_u8(auth_mode);
        match compress_openai_responses_live_zone(body.as_bytes(), auth, &model_str) {
            Ok(o) => outcome_to_result(&body, o),
            Err(e) => err_result_with_tokens(e.to_string(), count_tokens(&body)),
        }
    }) {
        Ok(r) => r,
        Err(_) => err_result("panic"),
    };
    make_result(result)
}

/// Replace the active CCR backend with a SQLite-backed store at the given
/// path. Idempotent — a second call with the same path overwrites the first.
/// `ttl_seconds` controls how long entries live; pass 0 to use the headroom
/// default (300s).
///
/// Returns NULL on success, or a heap-allocated error message on failure.
/// Free the returned pointer with `headroom_result_free`.
///
/// # Safety
/// `path` must be a valid null-terminated UTF-8 string. Must be called before
/// any compression that needs persistence; in-flight compressions still
/// holding an `Arc` to the previous store will continue using it until they
/// complete.
#[no_mangle]
pub unsafe extern "C" fn headroom_ccr_init_sqlite(
    path: *const c_char,
    ttl_seconds: u64,
) -> *mut c_char {
    let result: Result<(), String> = catch_unwind(|| {
        let path_str = match ptr_to_string(path) {
            Ok(p) => p,
            Err(r) => return Err(r.error.unwrap_or_else(|| "bad path".to_string())),
        };
        let ttl = if ttl_seconds == 0 { 300 } else { ttl_seconds };
        let store = SqliteCcrStore::open(&path_str, ttl).map_err(|e| e.to_string())?;
        let new_arc: Arc<dyn CcrStore> = Arc::new(store);
        match CCR_STORE.write() {
            Ok(mut guard) => {
                *guard = new_arc;
                Ok(())
            }
            Err(_) => Err("ccr store rwlock poisoned".to_string()),
        }
    })
    .unwrap_or_else(|_| Err("panic in ccr init".to_string()));
    match result {
        Ok(()) => std::ptr::null_mut(),
        Err(msg) => make_c_string(&msg),
    }
}

/// Replace the active CCR backend with a Redis-backed store. Optional
/// `key_prefix` is the Redis key namespace (NULL = headroom default).
/// `ttl_seconds=0` uses the headroom default (300s).
///
/// Returns NULL on success, or a heap-allocated error message on failure.
/// Free returned pointer with `headroom_result_free`.
///
/// # Safety
/// `url` must be a valid null-terminated UTF-8 string. `key_prefix` may be
/// NULL or a valid null-terminated UTF-8 string. Must be called before any
/// compression that needs the new backend; in-flight compressions still
/// holding an Arc to the previous store continue using it until they
/// complete.
#[no_mangle]
pub unsafe extern "C" fn headroom_ccr_init_redis(
    url: *const c_char,
    key_prefix: *const c_char,
    ttl_seconds: u64,
) -> *mut c_char {
    let result: Result<(), String> = catch_unwind(|| {
        let url_str = match ptr_to_string(url) {
            Ok(s) => s,
            Err(r) => return Err(r.error.unwrap_or_else(|| "bad url".to_string())),
        };
        let ttl = if ttl_seconds == 0 { 300 } else { ttl_seconds };
        let store_res = if key_prefix.is_null() {
            RedisCcrStore::open(&url_str, ttl)
        } else {
            match ptr_to_string(key_prefix) {
                Ok(p) => RedisCcrStore::open_with_prefix(&url_str, p, ttl),
                Err(r) => return Err(r.error.unwrap_or_else(|| "bad key_prefix".to_string())),
            }
        };
        let store = store_res.map_err(|e| e.to_string())?;
        let new_arc: Arc<dyn CcrStore> = Arc::new(store);
        match CCR_STORE.write() {
            Ok(mut guard) => {
                *guard = new_arc;
                Ok(())
            }
            Err(_) => Err("ccr store rwlock poisoned".to_string()),
        }
    })
    .unwrap_or_else(|_| Err("panic in redis init".to_string()));
    match result {
        Ok(()) => std::ptr::null_mut(),
        Err(msg) => make_c_string(&msg),
    }
}

/// Free a `*mut c_char` previously returned by any `headroom_*` FFI entry
/// point. Callers MUST use this instead of libc `free` — Rust's allocator is
/// not guaranteed to be ABI-compatible with the C runtime's free across all
/// platforms (musl, Windows, custom global_allocator).
///
/// # Safety
/// `ptr` must be a pointer returned by a headroom FFI call and not yet freed.
/// Passing a null pointer is a no-op. Double free is UB.
#[no_mangle]
pub unsafe extern "C" fn headroom_result_free(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    drop(CString::from_raw(ptr));
}

// ---------- Internal ----------

/// Read a null-terminated C string into an owned `String`. Copying decouples
/// the returned data from the caller's buffer lifetime, eliminating any
/// risk of UAF if downstream code retains the value.
///
/// # Safety
/// `ptr` must be NULL or a valid null-terminated UTF-8 buffer.
unsafe fn ptr_to_string(ptr: *const c_char) -> Result<String, CompressResult> {
    if ptr.is_null() {
        return Err(err_result("null pointer"));
    }
    CStr::from_ptr(ptr)
        .to_str()
        .map(|s| s.to_owned())
        .map_err(|_| err_result("invalid utf8"))
}

/// Read `len` bytes from `ptr` into an owned `String`, validating UTF-8.
/// `(NULL, 0)` is accepted as an empty body.
///
/// # Safety
/// `ptr` must be NULL or point to at least `len` valid bytes.
unsafe fn ptr_to_string_bytes(ptr: *const u8, len: usize) -> Result<String, CompressResult> {
    if len == 0 {
        return Ok(String::new());
    }
    if ptr.is_null() {
        return Err(err_result("null body pointer"));
    }
    let slice = std::slice::from_raw_parts(ptr, len);
    std::str::from_utf8(slice)
        .map(|s| s.to_owned())
        .map_err(|_| err_result("invalid utf8 in body"))
}

fn make_c_string(s: &str) -> *mut c_char {
    // CString::new errors on interior NUL. JSON output should never contain
    // raw NUL bytes — if it does, that signals an upstream bug and we surface
    // it as a parseable error envelope rather than silently stripping.
    match CString::new(s) {
        Ok(c) => c.into_raw(),
        Err(_) => CString::new(r#"{"modified":false,"body":null,"tokens_before":0,"tokens_after":0,"tokens_saved":0,"compression_ratio":1.0,"error":"interior_nul_in_ffi_output"}"#)
            .expect("static literal has no NUL")
            .into_raw(),
    }
}
