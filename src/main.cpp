#include <iostream>
#include "llm/llm.hpp"
#include <memory>
#include <libtcc.h>

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

    // Generate based on prompt
    auto gen_output = llm->prompt(llm_prompt).value();
    // Discard think tags + newline
    const std::string think_end = "</think>";
    const size_t tag_pos = gen_output.find(think_end);
    if (tag_pos != std::string::npos) {
        size_t start = tag_pos + think_end.size();
        while (start < gen_output.size() &&
                (gen_output[start] == '\n' || gen_output[start] == '\r')) {
            ++start;
        }
        gen_output.erase(0, start);
    }

    #ifndef NO_SAFETY_CHECK
    std::cout << "!!!Generated!!!" << std::endl;
    std::cout << gen_output << std::endl;

    std::cout << "!!!This is your one and only chance to check the C code that was emitted!!!" << std::endl;
    std::cin.get();
    #endif

    // good luck & have fun
    TCCState * tcc_state = tcc_new();
    if (!tcc_state) {
        std::cerr << "Couldn't start TCC, giving up." << std::endl;
        return 1;
    }

    tcc_set_options(tcc_state, "-I/usr/include");
    tcc_set_output_type(tcc_state, TCC_OUTPUT_MEMORY);

    // technically, we could send this back to the model for iterative improvement. perhaps next time.
    if (tcc_compile_string(tcc_state, gen_output.c_str()) == -1) {
        std::cerr << "Compile failed." << std::endl;
        return 1;
    }
    tcc_relocate(tcc_state);

    int (*new_main)(int, char**) = reinterpret_cast<int (*)(int, char**)>(tcc_get_symbol(tcc_state, "main"));
    char program_name[] = "chim";
    char* argv_no_args[] = { program_name, nullptr };
    new_main(1, argv_no_args);

    return 0;
}
