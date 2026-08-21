# chim

This is a Self-Modifying JIT LLM, just as a PoC.

## What does it do?

chim embeds an LLM along with llama.cpp and libtcc into a single program. A prompt to one-shot generate a program is fed into the LLM; the output of the LLM is JIT compiled into a valid binary inside the same program's space. We then execute the new program directly from memory.

## Why?

This program could be extended to behave as an obfuscator where the target code does not exist before runtime. In addition, this is a proto-"LLM-to-binary" system (that people keep shilling on twitter).

## Risks?
It is not recommended to run this software outside of a VM. The risk of generating bad code is extraordinarily high; you have been warned.

## How do I build this?

Create a model.gguf in the project root with a model of your choice.

```
mkdir build
cd build
cmake ..
```
## Limitations
Unfortunately, the model you use can be a little shit and disobey the system prompt for formatting. This can break the compiler, in which case your only solution is to rerun. Nothing can be done to fix this.

## Configurables
Define SHUT_UP_LLAMA in llm.cpp to disable log output.

Define NO_SAFETY_CHECK in main.cpp to disable safety checks (valid C code will be executed instantly).
