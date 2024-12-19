import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class HybridChatbot:
    def __init__(self, rag_model, finetuned_model_path):
        self.rag_model = rag_model

        # Cargar modelo fine-tuneado
        print("Cargando modelo fine-tuneado para modo híbrido...")
        self.ft_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
        self.ft_model = AutoModelForCausalLM.from_pretrained(
            finetuned_model_path,
            torch_dtype=torch.float16
        ).to("cuda")

    def generate_answer(self, query):
        # Primero RAG para obtener contexto
        query_emb = self.rag_model.embed(query)
        similarities = torch.nn.functional.cosine_similarity(query_emb, self.rag_model.embeddings, dim=-1)
        top_indices = torch.topk(similarities, k=5).indices.cpu().numpy()
        retrieved_contexts = self.rag_model.get_contexts(top_indices)

        prompt = self.build_prompt(query, retrieved_contexts)
        print(f"Prompt Híbrido:\n{prompt}")

        # Generar respuesta con modelo fine-tuneado
        return self._model_generate(prompt)

    def build_prompt(self, query, contexts):
        context_str = "\n---\n".join(contexts)
        return (
            f"El usuario pregunta: '{query}'\n"
            f"A continuación tienes información de películas relevantes:\n"
            f"{context_str}\n\n"
            f"Utiliza la información para responder al usuario de forma útil:\n"
            f"Respuesta:"
        )

    def _model_generate(self, prompt):
        inputs = self.ft_tokenizer(prompt, return_tensors="pt").to("cuda")
        output = self.ft_model.generate(
            **inputs,
            max_length=200,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=self.ft_tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
        answer = self.ft_tokenizer.decode(output[0], skip_special_tokens=True)
        return answer.strip()
