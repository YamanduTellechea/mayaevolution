import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="resource_tracker")


import gc

def clean_resources():
    print("Limpiando recursos...")
    gc.collect()
    torch.cuda.empty_cache()  # Si estás usando GPU (opcional)
    print("Recursos limpiados.")


class RAGChatbot:

    
    def __init__(self, index_path, embedding_model_name="all-MiniLM-L6-v2"):
        """
        Inicialización del chatbot RAG con un modelo de generación y un índice FAISS.
        """
        # Cargar el índice FAISS
        print(f"Cargando índice FAISS desde {index_path}...")
        try:
            self.index = faiss.read_index(index_path)
            print("Índice FAISS cargado correctamente.")
        except Exception as e:
            print(f"Error al cargar el índice FAISS: {e}")
            raise

        # Cargar el modelo de embeddings (para búsqueda semántica)
        print(f"Cargando modelo de embeddings {embedding_model_name}...")
        # self.embedding_model = SentenceTransformer(embedding_model_name)

        self.embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2", device="cpu")
        

        # Cargar el modelo de generación de texto (BLOOM)
        print("Cargando modelo de generación (bigscience/bloom-560m)...")
        self.tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
        self.model = AutoModelForCausalLM.from_pretrained(
            "bigscience/bloom-560m",
            device_map="auto",          # Deja que Accelerate maneje los dispositivos
            offload_folder="offload",   # Carpeta para offloading en disco
            offload_state_dict=True,    # Activa offloading para el estado del modelo
            low_cpu_mem_usage=True      # Reduce el uso de memoria en CPU
        )     # Mover explícitamente a CPU



    def generate_answer(self, query: str):
        """
        Genera una respuesta basada en el contexto recuperado desde FAISS.
        """
        print(f"Generando embeddings para la consulta: {query}")
        query_emb = self.embed(query)

        # Realizar la búsqueda en el índice FAISS
        k = 5
        print(f"Buscando en el índice FAISS los {k} contextos más cercanos...")
        D, I = self.index.search(query_emb, k)
        retrieved_contexts = self.get_contexts(I)

        # Construir el prompt para el modelo de generación
        prompt = self.build_prompt(query, retrieved_contexts)
        print(f"Prompt construido:\n{prompt}")

        # Generar la respuesta usando el modelo
        print("Generando respuesta con el modelo de lenguaje...")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(
            **inputs,
            max_length=100,
            do_sample=True,
            temperature=0.7
        )
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return answer

    def embed(self, text):
        """
        Convierte el texto en un embedding utilizando el modelo de embeddings.
        """
        print("Generando embeddings para el texto...")
        emb = self.embedding_model.encode(text, convert_to_numpy=True).reshape(1, -1)
        return emb.astype('float32')

    def get_contexts(self, indices):
        """
        Recupera los textos asociados a los índices devueltos por FAISS.
        """
        print(f"Recuperando contextos para índices: {indices}")
        # Placeholder: Aquí deberías mapear índices a tus textos reales.
        contexts = [f"Documento recuperado {i}" for i in indices[0]]
        return contexts

    def build_prompt(self, query, contexts):
        """
        Construye el prompt para el modelo de generación usando el query y los contextos recuperados.
        """
        print("Construyendo prompt para el modelo...")
        context_str = "\n".join(contexts)
        return f"Context:\n{context_str}\n\nUser: {query}\nAssistant:"


if __name__ == "__main__":
    try:
        chatbot = RAGChatbot(index_path="embeddings/faiss_index.bin")
        query = "Tell me about the moon landing."
        answer = chatbot.generate_answer(query)
        print("Respuesta del chatbot:", answer)
    finally:
        clean_resources()


