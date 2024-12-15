from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "bigscience/bloom-560m"  # Alternativa más ligera y pública
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Guarda el modelo localmente
tokenizer.save_pretrained("./bigscience/bloom-560m")
model.save_pretrained("./bigscience/bloom-560m")