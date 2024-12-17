from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .rag_pipeline import RAGChatbot
# from .fine_tune_pipeline import FineTunedChatbot
# from .hybrid_pipeline import HybridChatbot

# Quita la inicialización global
rag_model = None
ft_model = None
hb_model = None

class ChatView(APIView):
    def post(self, request, format=None):
        user_query = request.data.get("query", "")
        mode = request.data.get("mode", "rag")  # 'rag', 'finetune', 'hybrid'

        # Inicializar las instancias en el momento que se necesitan
        global rag_model, ft_model, hb_model
        if rag_model is None:
            dataset_path = "chatapp/dataset"
            embedding_file = "embeddings.npz"
            rag_model = RAGChatbot(dataset_path, embedding_file)
        # if ft_model is None:
        #     from .fine_tune_pipeline import FineTunedChatbot
        #     ft_model = FineTunedChatbot(model_path="path/to/your/llama2-finetuned-model")
        # if hb_model is None:
        #     from .hybrid_pipeline import HybridChatbot
        #     hb_model = HybridChatbot(rag_model=rag_model, ft_model=ft_model)

        if mode == "rag":
            answer = rag_model.generate_answer(user_query)
        # elif mode == "finetune":
        #     answer = ft_model.generate_answer(user_query)
        # elif mode == "hybrid":
        #     answer = hb_model.generate_answer(user_query)
        else:
            return Response({"error": "Modo no válido"}, status=status.HTTP_400_BAD_REQUEST)

        return Response({"query": user_query, "answer": answer}, status=status.HTTP_200_OK)


class ResultsView(APIView):
    def get(self, request, format=None):
        results = {
            "comparisons": [
                {"input": "Hello!", "RAG": "Hi there, how can I help?", "FineTune": "Hello, what would you like to discuss?", "Hybrid": "Hello! How can I assist?"}
            ],
            "metrics": {
                "BLEU": {"RAG": 0.25, "FineTune": 0.23, "Hybrid": 0.27},
                "CostEstimate": {"RAG": "Low", "FineTune": "Medium", "Hybrid": "Medium-Low"}
            }
        }
        return Response(results, status=status.HTTP_200_OK)

