from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class FineTunedChatbot:
    def __init__(self, model_path):
        # Carga del modelo Llama2 fine-tuneado localmente
        # Ejemplo: "path/to/llama2-finetuned" debe contener la carpeta del modelo
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    def generate_answer(self, query: str):
        # Ajustar el prompt según Llama2 chat style
        prompt = f"<s>[INST] <<SYS>>\nEres un asistente útil y amable.\n<</SYS>>\n{query}[/INST]"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_length=200, do_sample=True, temperature=0.7)
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # Llama2 chat format: filtrar el prompt, quedarte con la respuesta.
        # Dependiendo de cómo hayas hecho el fine-tuning, ajusta el parseo.
        # Supongamos que la respuesta es lo que viene después del prompt
        # Aquí simplificamos y devolvemos todo excepto el prompt original.
        # En producción: filtra adecuadamente.
        return answer.replace(prompt, "").strip()
