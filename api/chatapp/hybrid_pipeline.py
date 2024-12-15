class HybridChatbot:
    def __init__(self, rag_model, ft_model):
        self.rag_model = rag_model
        self.ft_model = ft_model

    def generate_answer(self, query: str):
        # Primero obtenemos contexto con RAG
        context = self.rag_model.generate_answer(query)
        # Después pasamos todo al modelo fine-tuneado para refinar
        refined_query = f"Por favor, mejora esta respuesta teniendo en cuenta el contexto adicional:\nContexto:\n{context}\n\nPregunta: {query}\n"
        return self.ft_model.generate_answer(refined_query)
