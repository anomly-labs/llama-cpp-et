#include "arg.h"
#include "common.h"
#include "log.h"

#include "cli-context.h"
#include "ggml-backend.h"

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>

#include <signal.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void signal_handler(int) {
    if (cli_context::interrupted().load()) {
        // second Ctrl+C - exit immediately
        // make sure to clear colors before exiting (not using LOG or console.cpp here to avoid deadlock)
        fprintf(stdout, "\033[0m\n");
        fflush(stdout);
        std::exit(130);
    }
    cli_context::interrupted().store(true);
}
#endif

// satisfies -Wmissing-declarations
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
    static const char * mm[] = { "ffn_norm-", "ffn_gate-", "ffn_up-", "ffn_swiglu-", "ffn_gate_par-",
                                 "ffn_out-", "attn_norm-", "Vcur-", "kqv_out-", "attn_out-", nullptr };
    bool is_mm = false;
    if (matmuls) {
        for (int i = 0; mm[i]; i++) {
            if (strncmp(t->name, mm[i], strlen(mm[i])) == 0) { is_mm = true; break; }
        }
        // Qcur/Kcur are re-emitted under the same name after RoPE; only the MUL_MAT output
        // is the exact unit. Tag those "Qcur_mm-<il>" / "Kcur_mm-<il>" in the dump.
        if (!is_mm && (t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_ROPE) &&
            (strncmp(t->name, "Qcur-", 5) == 0 || strncmp(t->name, "Kcur-", 5) == 0)) {
            is_mm = true;   // post-RoPE rows are tagged Qcur_rope-<il> (deterministic RoPE gate)
        }
    }
    const bool wanted = strcmp(t->name, "result_norm") == 0 || strcmp(t->name, "result_output") == 0
                     || (layers && strncmp(t->name, "l_out-", 6) == 0) || is_mm;
    if (ask) {
        return wanted;
    }
    if (!wanted || t->type != GGML_TYPE_F32) {
        return true;
    }
    const int64_t n = t->ne[0];
    const int64_t row = t->ne[1] - 1;                   // last position = the sampled token
    std::vector<float> buf((size_t) n);
    ggml_backend_tensor_get(t, buf.data(), (size_t) row * t->nb[1], (size_t) n * sizeof(float));
    FILE * f = fopen((const char *) user_data, "ab");
    if (!f) {
        return true;
    }
    char tname[GGML_MAX_NAME + 8];
    if (t->op == GGML_OP_MUL_MAT && (strncmp(t->name, "Qcur-", 5) == 0 || strncmp(t->name, "Kcur-", 5) == 0)) {
        snprintf(tname, sizeof(tname), "%.4s_mm-%s", t->name, t->name + 5);   // Qcur_mm-<il>
    } else if (t->op == GGML_OP_ROPE && (strncmp(t->name, "Qcur-", 5) == 0 || strncmp(t->name, "Kcur-", 5) == 0)) {
        snprintf(tname, sizeof(tname), "%.4s_rope-%s", t->name, t->name + 5); // Qcur_rope-<il>
    } else {
        snprintf(tname, sizeof(tname), "%s", t->name);
    }
    if (t->op == GGML_OP_ROPE && t->src[1] && t->src[1]->type == GGML_TYPE_I32) {
        // the position of the dumped row (src[1] = positions), so a verifier can re-execute RoPE
        int32_t pos = -1;
        const int64_t prow = row < t->src[1]->ne[0] ? row : t->src[1]->ne[0] - 1;
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

int llama_cli(int argc, char ** argv);

int llama_cli(int argc, char ** argv) {
    common_params params;

    params.verbosity = LOG_LEVEL_ERROR; // by default, less verbose logs

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CLI)) {
        return 1;
    }
    if (const char * logits_out = getenv("INVAR_LOGITS_OUT")) {
        params.cb_eval           = invar_logits_cb;
        params.cb_eval_user_data = (void *) logits_out;
    }

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    cli_context ctx_cli(params);

    if (!ctx_cli.init()) {
        return 1;
    }

    return ctx_cli.run();
}
