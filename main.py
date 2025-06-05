#!/usr/bin/env python
# coding: utf-8

import torch
from transformers import (  # Still needed for apply_prompt_template or direct use if any
    AutoModelForCausalLM,
    AutoTokenizer,
)

from proxy_model import ProxyModel  # Import the ProxyModel from the new file

# 1. Choose a small pre-trained model
# 'gpt2' is a good small model for demonstration purposes.
# Other small options could be 'gpt2-medium', 'distilgpt2', 'sshleifer/tiny-gpt2', etc.
# model_name = "gpt2"
model_name = "Qwen/Qwen3-0.6B"  # This will be the actual model loaded by ProxyModel

# ProxyModel definition is now in proxy_model.py

print(f"Loading tokenizer for '{model_name}'...")
# 2. Load the tokenizer
# The tokenizer is responsible for converting text into token IDs that the model understands.
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Instantiating ProxyModel with actual model '{model_name}'...")
# 3. Load the model (using ProxyModel)
# The ProxyModel will internally load AutoModelForCausalLM.from_pretrained(model_name)
model = ProxyModel(actual_model_name_or_path=model_name)

print("ProxyModel instantiated successfully, real model loaded internally.")


# --- Template Formatting ---
def apply_prompt_template(
    prompt, model_name, system_prompt="You are a helpful assistant."
):
    """Applies a model-specific template to the prompt, including a system prompt."""
    if "Qwen" in model_name:  # Example for Qwen models
        # Qwen models often use a specific chat format.
        # Refer to official Qwen documentation for precise formatting.
        # Example: <|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
        formatted_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    elif "gpt2" in model_name:  # Example for GPT-2
        # GPT-2 doesn't have a strict template, but you can prepend system and user roles.
        formatted_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
    elif "Llama" in model_name:  # Example for Llama models (conceptual)
        # Llama models (like Llama-2-chat, Llama-3-instruct) have specific chat templates.
        # Example for Llama-2-chat / Llama-3-instruct:
        # <s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST]
        # Note: Llama-3 uses a different template structure, often with <|begin_of_text|>, <|start_header_id|>, etc.
        # This is a simplified Llama-2 style example.
        formatted_prompt = (
            f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
        )
    else:
        # Default: no specific template, use prompt as is, optionally prepend system prompt if provided
        print(
            f"Warning: No specific template found for model '{model_name}'. Using raw prompt with optional system prompt."
        )
        if system_prompt:
            formatted_prompt = f"System: {system_prompt}\nUser: {prompt}"
        else:
            formatted_prompt = prompt
    return formatted_prompt


# --- End Template Formatting ---

# Set the pad_token_id to eos_token_id for GPT-like models that don't have a dedicated pad token
# This is important for the generate method, especially if handling batches (though not strictly needed for a single input)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 4. Define your input prompt
# raw_prompt = "What is the capital of France?"
# print(f"\nRaw Input Prompt: '{raw_prompt}'")

# Apply the template
# You can customize the system_prompt here if needed for a specific run
chat = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is the capital of France?"},
    # {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
]

# formatted_prompt = apply_prompt_template(
#     raw_prompt, model_name, system_prompt=system_message
# )
formatted_prompt = tokenizer.apply_chat_template(
    chat, tokenize=False, add_generation_prompt=True
)
print(f"Formatted Prompt for LLM: '{formatted_prompt}'")

# 5. Tokenize the input prompt
# We convert the prompt string into a sequence of token IDs.
# return_tensors='pt' ensures we get PyTorch tensors.
input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt")

print(f"Tokenized Input IDs: {input_ids}")

# 6. Generate text
# Use the model's generate method to produce output tokens.
# max_length: Maximum total length of the generated sequence (prompt + new tokens).
# num_return_sequences: How many different sequences to generate.
# no_repeat_ngram_size: Prevent the model from repeating n-grams (helps with fluency).
print("\nGenerating text...")
output_sequences = model.generate(
    input_ids=input_ids,
    max_length=len(input_ids[0]) + 256,  # Generate prompt length + 30 new tokens
    num_return_sequences=1,
    no_repeat_ngram_size=2,  # Example parameter to control generation
    pad_token_id=tokenizer.pad_token_id,  # Use the specified pad token ID
    eos_token_id=tokenizer.eos_token_id,  # Use the specified eos token ID
)
print("Text generation complete.")

# 7. Decode the output tokens back into text
# We get the first (and only) generated sequence [0]
# skip_special_tokens=True removes special tokens like PAD or EOS from the output string.
generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=False)

# 8. Print the generated text
print("\n--- Generated Text ---")
print(generated_text)
print("--------------------")


# In[2]:


# type(model)


# In[3]:


# model


# In[9]:


# get_ipython().run_line_magic('pinfo', 'model.generate')
