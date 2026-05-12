// headroom.h - C FFI bindings for headroom-core compression.
// Generated manually based on lib.rs signatures.
#ifndef __HEADROOM_H__
#define __HEADROOM_H__

#include <stdint.h>
#include <stddef.h>

// Auth modes (match headroom-core AuthMode enum order)
#define HEADROOM_AUTH_PAYG 0
#define HEADROOM_AUTH_OAUTH 1
#define HEADROOM_AUTH_SUBSCRIPTION 2
#define HEADROOM_AUTH_UNKNOWN 3

// Compress OpenAI /v1/chat/completions body.
// body: JSON bytes (not null-terminated)
// body_len: byte length of body
// model: null-terminated model name
// auth_mode: one of HEADROOM_AUTH_* constants
// Returns: JSON result string, owned by Rust. Caller must NOT free.
const char* headroom_compress_openai(
    const uint8_t* body,
    size_t body_len,
    const char* model,
    uint8_t auth_mode
);

// Compress Anthropic /v1/messages body.
// frozen_count: number of pinned messages from conversation start.
const char* headroom_compress_anthropic(
    const uint8_t* body,
    size_t body_len,
    size_t frozen_count,
    const char* model,
    uint8_t auth_mode
);

// Retrieve original content from CCR store by hash.
// Returns: "OK:<content>" or "ERR:<json>"
const char* headroom_ccr_get(const char* hash);

// Free a result string allocated by Rust (no-op, kept for API compat).
void headroom_result_free(const char* ptr);

#endif // __HEADROOM_H__
