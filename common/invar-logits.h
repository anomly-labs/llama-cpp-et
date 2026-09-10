// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// invar-logits.h — the INVAR whole-graph dump hook (INVAR_LOGITS_OUT / _LAYERS / _MATMULS), shared by
// llama-cli and llama-batched. Header-only; include from one translation unit per binary.
#pragma once
#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Anomly INVAR client-side spot-check (CSC) hook. INVAR_LOGITS_OUT=<file> appends, for
// every graph evaluation, the LAST ROW of the final-norm hidden state ("result_norm",
// what the lm_head matmul consumes) and of the logits ("result_output") as JSON lines:
//   {"tensor":"result_norm","n":576,"row":0,"hex":"<f32 little-endian hex>"}
// A verifier holding the GGUF re-quantises the hidden row exactly as ggml does and
// re-executes sampled lm_head rows in the exact quire; under the b-posit8 profile the
// re-executed logits must match these bit for bit (tests/csc/csc_verify.py).
// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// INVAR_LOGITS_LAYERS=1 additionally captures every layer's residual-stream output
// ("l_out-<layer>", last row) so a verifier with the same deployment can localise a
// divergence to a layer. Those rows are NOT cross-implementation re-executable today
// (RMSNorm, RoPE, SiLU and softmax run in float32 in the graph); see docs/SPOT-CHECK.md.
static bool invar_logits_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    static int layers = -1, matmuls = -1;
    if (layers < 0) { const char * ev = getenv("INVAR_LOGITS_LAYERS"); layers = (ev && ev[0] == '1') ? 1 : 0; }
    if (matmuls < 0) { const char * ev = getenv("INVAR_LOGITS_MATMULS"); matmuls = (ev && ev[0] == '1') ? 1 : 0; }
    // INVAR_LOGITS_MATMULS=1: the inputs and outputs of every FFN / attention-output matmul
    // (the exact units) so a verifier can re-execute sampled rows of each with the GGUF
    // weights: ffn_norm -> ffn_gate, ffn_up ; ffn_swiglu|ffn_gate_par -> ffn_out ;
    // attn_norm -> Vcur ; kqv_out -> attn_out.
    static const char * mm[] = { "ffn_norm-", "ffn_gate-", "ffn_up-", "ffn_swiglu-", "ffn_gate_par-", "ffn_geglu-",
                                 "ffn_out-", "attn_norm-", "Vcur-", "kqv_out-", "attn_out-",
                                 "Qcur_normed-", "Kcur_normed-", "attn_post_norm-", "sa_out-", "ffn_post_norm-",   // gemma3
                                 nullptr };
    bool is_mm = false;
    if (matmuls) {
        for (int i = 0; mm[i]; i++) {
            if (strncmp(t->name, mm[i], strlen(mm[i])) == 0) { is_mm = true; break; }
        }
        // Qcur/Kcur are re-emitted under the same name after RoPE; only the MUL_MAT output
        // is the exact unit. Tag those "Qcur_mm-<il>" / "Kcur_mm-<il>" in the dump.
        if (!is_mm && (t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_ROPE || t->op == GGML_OP_ADD) &&
            (strncmp(t->name, "Qcur-", 5) == 0 || strncmp(t->name, "Kcur-", 5) == 0)) {
            is_mm = true;   // post-RoPE rows are tagged Qcur_rope-<il>, post-bias rows Qcur_bias-<il>
        }
        // the attention output projection may be unnamed (gemma3): it is the matmul fed by kqv_out-<il>
        if (!is_mm && t->op == GGML_OP_MUL_MAT && t->src[1] && strncmp(t->src[1]->name, "kqv_out-", 8) == 0
            && strncmp(t->name, "attn_out-", 9) != 0) {
            is_mm = true;
        }
    }
    static int names = -1;   // INVAR_LOGITS_NAMES=1: list every graph tensor (name, op, dims) once on stderr
    if (names < 0) { const char * ev = getenv("INVAR_LOGITS_NAMES"); names = (ev && ev[0] == '1') ? 1 : 0; }
    if (names == 1 && ask) {
        fprintf(stderr, "[invar-names] %s op=%s ne=%lld,%lld,%lld type=%s\n", t->name, ggml_op_name(t->op),
                (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], ggml_type_name(t->type));
    }
    const bool wanted = strcmp(t->name, "result_norm") == 0 || strcmp(t->name, "result_output") == 0
                     || (layers && strncmp(t->name, "l_out-", 6) == 0) || is_mm
                     || (matmuls && (strcmp(t->name, "inp_embd") == 0 || (strcmp(t->name, "embd") == 0 && t->op == GGML_OP_GET_ROWS)))   // layer-0 residual input
                     || (matmuls && strcmp(t->name, "inp_tokens") == 0)
                     || (matmuls && strcmp(t->name, "inp_scaled") == 0);   // gemma: embedding * sqrt(n_embd)
    if (ask) {
        return wanted;
    }
    if (matmuls && !ask && t->op == GGML_OP_GET_ROWS && (strcmp(t->name, "embd") == 0 || strcmp(t->name, "inp_embd") == 0)
        && t->src[1] && t->src[1]->type == GGML_TYPE_I32) {
        // the token ids of this evaluation (prefill: the prompt; decode: the sampled token) from the
        // embedding lookup's index tensor (input leaves never reach the callback), so a reference
        // re-executor can reproduce the whole sequence without llama.cpp's tokenizer
        const int64_t nt = t->src[1]->ne[0];
        std::vector<int32_t> ids((size_t) nt);
        ggml_backend_tensor_get(t->src[1], ids.data(), 0, (size_t) nt * sizeof(int32_t));
        FILE * f = fopen((const char *) user_data, "ab");
        if (f) {
            fprintf(f, "{\"tensor\":\"inp_tokens\",\"n\":%lld,\"ids\":[", (long long) nt);
            for (int64_t i = 0; i < nt; i++) fprintf(f, "%s%d", i ? "," : "", ids[i]);
            fputs("]}\n", f);
            fclose(f);
        }
        // fall through: the embedding row itself is dumped below
    }
    if (!wanted || t->type != GGML_TYPE_F32) {
        return true;
    }
    // a prompt chunk that produces no logits leaves the post-selection tensors of the last layer
    // and the final norm/logits EMPTY (0 rows): keep the evaluation boundary with an n=0 line for
    // the logits and skip the rest (the row index would be -1).
    if (t->ne[1] == 0 || t->ne[0] == 0 || (t->ne[2] == 0)) {
        if (strcmp(t->name, "result_output") == 0) {
            FILE * f = fopen((const char *) user_data, "ab");
            if (f) { fprintf(f, "{\"tensor\":\"result_output\",\"n\":0,\"row\":0,\"hex\":\"\"}\n"); fclose(f); }
        }
        return true;
    }
    // last position = the sampled token. RoPE outputs are 3-D [head_dim, n_head, n_tokens]:
    // dump every head of the last token (ne0*ne1 values) so the row matches the matmul row.
    const bool per_head = strstr(t->name, "_normed") != NULL;              // [head_dim, n_head, n_tokens] even for one token
    const bool three_d = t->op == GGML_OP_ROPE || t->ne[2] > 1 || (per_head && t->ne[1] > 1);
    const int64_t n   = three_d ? t->ne[0] * t->ne[1] : t->ne[0];
    const int64_t row = three_d ? t->ne[2] - 1 : t->ne[1] - 1;
    std::vector<float> buf((size_t) n);
    ggml_backend_tensor_get(t, buf.data(), (size_t) row * (three_d ? t->nb[2] : t->nb[1]), (size_t) n * sizeof(float));
    FILE * f = fopen((const char *) user_data, "ab");
    if (!f) {
        return true;
    }
    char tname[GGML_MAX_NAME + 8];
    if (t->op == GGML_OP_MUL_MAT && (strncmp(t->name, "Qcur-", 5) == 0 || strncmp(t->name, "Kcur-", 5) == 0)) {
        snprintf(tname, sizeof(tname), "%.4s_mm-%s", t->name, t->name + 5);   // Qcur_mm-<il>
    } else if (t->op == GGML_OP_ROPE && (strncmp(t->name, "Qcur-", 5) == 0 || strncmp(t->name, "Kcur-", 5) == 0)) {
        snprintf(tname, sizeof(tname), "%.4s_rope-%s", t->name, t->name + 5); // Qcur_rope-<il>
    } else if (t->op == GGML_OP_ADD && (strncmp(t->name, "Qcur-", 5) == 0 || strncmp(t->name, "Kcur-", 5) == 0 || strncmp(t->name, "Vcur-", 5) == 0)) {
        snprintf(tname, sizeof(tname), "%.4s_bias-%s", t->name, t->name + 5); // Qcur_bias-<il> (projection bias added)
    } else if (t->op == GGML_OP_MUL_MAT && t->src[1] && strncmp(t->src[1]->name, "kqv_out-", 8) == 0 && strncmp(t->name, "attn_out-", 9) != 0) {
        snprintf(tname, sizeof(tname), "attn_out-%s", t->src[1]->name + 8);    // unnamed wo product (gemma3)
    } else {
        snprintf(tname, sizeof(tname), "%s", t->name);
    }
    if (t->op == GGML_OP_ROPE && t->src[1] && t->src[1]->type == GGML_TYPE_I32) {
        // the position of the dumped row (src[1] = positions), so a verifier can re-execute RoPE
        int32_t pos = -1;
        const int64_t prow = row < t->src[1]->ne[0] ? row : t->src[1]->ne[0] - 1;   // row = last token
        ggml_backend_tensor_get(t->src[1], &pos, (size_t) prow * sizeof(int32_t), sizeof(int32_t));
        fprintf(f, "{\"tensor\":\"%s\",\"n\":%lld,\"row\":%lld,\"pos\":%d,\"hex\":\"", tname, (long long) n, (long long) row, pos);
    } else {
        fprintf(f, "{\"tensor\":\"%s\",\"n\":%lld,\"row\":%lld,\"hex\":\"", tname, (long long) n, (long long) row);
    }
    const unsigned char * b = (const unsigned char *) buf.data();
    for (size_t i = 0; i < (size_t) n * sizeof(float); i++) {
        fputc("0123456789abcdef"[b[i] >> 4], f);
        fputc("0123456789abcdef"[b[i] & 15], f);
    }
    fputs("\"}\n", f);
    fclose(f);
    return true;
}
