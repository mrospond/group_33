from llama_cpp import Llama
model_path = "models/llama-2-7b.Q4_K_M.gguf"

# If you want to use larger models...
#model_path = "models/llama-2-13b.Q4_K_M.gguf"
# All models are available at https://huggingface.co/TheBloke. Make sure you download the ones in the GGUF format

question = "What is the capital of Italy? "
llm = Llama(model_path=model_path, verbose=False)
print("Asking the question \"%s\" to %s (wait, it can take some time...)" % (question, model_path))
output = llm(
      question, # Prompt
      max_tokens=32, # Generate up to 32 tokens
      stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
)
print("Here is the output")
print(output['choices'])