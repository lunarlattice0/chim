#include <iostream>
#include "llm/llm.hpp"
#include <memory>
#include <libtcc.h>

#define ITERATIVE_MODE

std::string stripThink(std::string original) {
    const std::string think_end = "</think>";
    const size_t tag_pos = original.find(think_end);
    if (tag_pos != std::string::npos) {
        size_t start = tag_pos + think_end.size();
        while (start < original.size() &&
                (original[start] == '\n' || original[start] == '\r')) {
            ++start;
        }
        original.erase(0, start);
    }
    return original;
}

void safetyCheck(const std::string& generated) {
    #ifndef NO_SAFETY_CHECK
    std::cout << "!!!Generated!!!" << std::endl;
    std::cout << generated << std::endl;

    std::cout << "!!!This is your one and only chance to check the C code that was emitted!!!" << std::endl;
    std::cin.get();
    #endif
}

int main(int argc, char* argv[]) {
    std::string llm_prompt;

    if (argc != 2) {
        std::cout << "usage: chim \"llm generation prompt\"" << std::endl;
        return 1;
    } else {
        llm_prompt = argv[1];
    }

    // start up llama.cpp
    std::unique_ptr<LLM> llm = std::make_unique<LLM>();

    // good luck & have fun
    TCCState * tcc_state = tcc_new();
    if (!tcc_state) {
        std::cerr << "Couldn't start TCC, giving up." << std::endl;
        return 1;
    }
    tcc_set_options(tcc_state, "-I/usr/include");
    tcc_set_output_type(tcc_state, TCC_OUTPUT_MEMORY);

    std::string err_buf;
    tcc_set_error_func(tcc_state, &err_buf, [](void * opaque, const char * msg) -> void{
        auto *buf = static_cast<std::string*>(opaque);
        *buf = msg;
    });

    bool compilePassed = false;
    do {
        // Generate based on prompt
        auto gen_output = llm->prompt(llm_prompt).value();
        // Discard think tags + newline
        auto cleaned_out = stripThink(gen_output);
        auto compile_state = tcc_compile_string(tcc_state, cleaned_out.c_str());

        if (compile_state == -1) {
            #ifndef ITERATIVE_MODE
                std::cerr << "Compile failed." << std::endl;
                return 1;
            #else
                llm_prompt = "Compile failed, with error(s): " + err_buf;
            #endif
        } else if (compile_state == 0) {
            compilePassed = true;
            safetyCheck(cleaned_out);
        }
    } while (!compilePassed);
    tcc_relocate(tcc_state);

    int (*new_main)(int, char**) = reinterpret_cast<int (*)(int, char**)>(tcc_get_symbol(tcc_state, "main"));
    char program_name[] = "chim";
    char* argv_no_args[] = { program_name, nullptr };
    new_main(1, argv_no_args);

    return 0;
}
