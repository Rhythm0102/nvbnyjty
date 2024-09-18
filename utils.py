from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text):
    return len(tokenizer.encode(text))