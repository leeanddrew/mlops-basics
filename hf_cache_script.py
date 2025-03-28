from transformers import AutoTokenizer, AutoModel

model_name = "google/bert_uncased_L-2_H-128_A-2"
cache_dir = "./hf_cache"

# Save tokenizer and model to local cache dir
AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
AutoModel.from_pretrained(model_name, cache_dir=cache_dir)