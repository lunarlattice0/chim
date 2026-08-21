#include "llm.hpp"
#include "ggml.h"
#include "llama.h"
#include <array>
#include <expected>
#include <stdexcept>
#include <string>
#include <iostream>

#define SHUT_UP_LLAMA

constexpr unsigned char mem_model[] = {
   #embed "../../model.gguf"
};

constexpr std::size_t model_size = sizeof(mem_model);

FILE * fp;

LLM::~LLM() {
    formatted.clear();
    messages.clear();

    llama_sampler_free(this->smpl);
    llama_free(this->ctx);
    llama_model_free(this->model);

    fclose(fp);
}

void LLM::reset() {
    formatted.clear();
    messages.clear();
    prev_len = 0;

    auto mem = llama_get_memory(this->ctx);
    llama_memory_clear(mem, true);
}

// This actually generates the text
std::expected<std::string, LLMFailure> LLM::generate(const std::string prompt) {
    std::string response;

    const bool is_first = llama_memory_seq_pos_max(llama_get_memory(this->ctx), 0) == -1;

    // tokenize
    const int n_prompt_tokens = -llama_tokenize(this->vocabSize, prompt.c_str(), prompt.size(), NULL, 0, is_first, true);
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(this->vocabSize, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) <0) {
        return std::unexpected(LLMFailure::tokenize_failure);
    }

    //prepare a batch
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    llama_token new_token_id;
    while (true) {
        int n_ctx = llama_n_ctx(ctx);
        int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
        if (n_ctx_used + batch.n_tokens > n_ctx) {
            return std::unexpected(LLMFailure::ctx_exceeded);
        }

        int ret = llama_decode(this->ctx, batch);
        if (ret != 0) {
            return std::unexpected(LLMFailure::decode_failure);
        }

        new_token_id = llama_sampler_sample(smpl, this->ctx, -1);

        if (llama_vocab_is_eog(this->vocabSize, new_token_id)) {
            break;
        }

        auto buf = std::array<char, 256>();
        int n = llama_token_to_piece(this->vocabSize, new_token_id, buf.data(), buf.size(), 0, true);
        if (n < 0) {
            return std::unexpected(LLMFailure::token_to_piece_failure);
        }
        std::string piece(buf.data(), n);
        response += piece;

        batch = llama_batch_get_one(&new_token_id, 1);
    }
    return response;
}

std::vector<std::string> msgBuf;
// This preformats the text
// built on the old exaigesis subsystem so prepare for funny business
std::expected<std::string, LLMFailure> LLM::prompt(const std::string input) {
    msgBuf.push_back(input);
    // TODO: Build custom chat template.
    const char * tmpl = llama_model_chat_template(this->model, nullptr);

    // Add system prompt
    this->messages.push_back({"system",
        "You are a C program generator. Respond only with the source code of a single, complete C program.\n"
        "- Output raw C code only. No greetings, explanations, summaries, or any other text.\n"
        "- Do not use Markdown or code fences; the entire response must be valid C source.\n"
        "- Make the program self-contained: include all necessary #include directives, "
        "a main() function, and use standard C (C11) with no external dependencies.\n"
        "- The main() function you create cannot take any arguments. Process user input and output with stdin and stdout.\n"
        "- Comments are allowed only inside the C code.\n"
        "- If a request cannot be fulfilled with a C program, respond with a single C comment stating why."
    });

    // Add request
    const char * lastMsg = msgBuf[msgBuf.size() - 1].c_str();
    this->messages.push_back({"user", lastMsg});

    // Apply template
    int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
    if (new_len > (int)formatted.size()) {
        formatted.resize(new_len);
        new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
    }

    if (new_len < 0) {
        return std::unexpected(LLMFailure::chat_template_failure);
    }

    // Compress the templated messages into a formatted single message for prompting.
    std::string prompt(formatted.begin() + prev_len, formatted.begin() + new_len);
    auto response = generate(prompt);
    if (!response) {
        return response; // Something broke in the prompt generation.
    }
    prev_len = new_len;

    return response.value();
}

LLM::LLM() {
    #ifdef SHUT_UP_LLAMA
    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        (void) level;
        (void) text;
        (void) user_data;
    }, NULL);
    #endif

    // create a FILE * from .rodata embed
    fp = fmemopen((void*)&mem_model[0], model_size, "r");
    if (fp == NULL) {
        GGML_ABORT("Couldn't create FP from model in memory.");
    }


    ggml_backend_load_all();

    this->mp = llama_model_default_params();
    this->mp.load_mode = LLAMA_LOAD_MODE_NONE;
    this->mp.n_gpu_layers = GPULAYERS;
    this->model = llama_model_load_from_file_ptr(fp, mp);
    if (!this->model)
        GGML_ABORT("Failed to load model");
    this->vocabSize = llama_model_get_vocab(model);

    this->ctx_params = llama_context_default_params();
    this->ctx_params.n_ctx = CTXSIZE;
    this->ctx_params.n_batch = CTXSIZE;

    this->ctx = llama_init_from_model(this->model, this->ctx_params);
    if (!this->ctx)
        GGML_ABORT("failed to create ctx");

    this->smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(1.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(20));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    this->formatted = std::vector<char>(llama_n_ctx(ctx));
}
