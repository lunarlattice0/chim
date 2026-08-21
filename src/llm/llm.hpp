#pragma once
#include "llama.h"
#include "ggml.h"
#include "llama.h"
#include <expected>
#include <vector>
#include <string>

#define GPULAYERS 0
#define CTXSIZE 16384

enum class LLMFailure {
    tokenize_failure,
    ctx_exceeded,
    decode_failure,
    token_to_piece_failure,
    chat_template_failure,
};


class LLM {
    public:
        LLM();
        std::expected<std::string, LLMFailure> prompt(const std::string input);
        void reset();
        ~LLM();
    private:
        std::expected<std::string, LLMFailure> generate(const std::string prompt);

        llama_model_params mp;
        llama_model * model;
        const llama_vocab * vocabSize;
        llama_context_params ctx_params;
        // TODO: User CTX switching.
        llama_context * ctx;
        llama_sampler * smpl;
        // TODO: Figure out how we want to schedule jobs.
        std::vector<llama_chat_message> messages;
        std::vector<char> formatted;
        int prev_len = 0;
};
