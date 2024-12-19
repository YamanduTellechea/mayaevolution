from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .rag_pipeline import RAGChatbot
# Solo descomenta si ya tienes modelos fine-tuneados
# from .fine_tune_pipeline import ...
from .hybrid_pipeline import HybridChatbot
import os

rag_model = None
ft_model = None
hb_model = None

class ChatView(APIView):
    def post(self, request, format=None):
        user_query = request.data.get("query", "")
        mode = request.data.get("mode", "rag")  # 'rag', 'finetune', 'hybrid'

        global rag_model, ft_model, hb_model

        if rag_model is None:
            dataset_path = "chatapp/dataset"
            embedding_file = "movie_embeddings.npz"
            rag_model = RAGChatbot(
                dataset_path=dataset_path, 
                embedding_file=embedding_file,
                embedding_model_name="sentence-transformers/all-distilroberta-v1",
                base_model_name="mistralai/Mistral-7B-Instruct-v0.3"
            )

        if mode == "rag":
            answer = rag_model.generate_answer(user_query)

        elif mode == "finetune":
            # Necesitas haber finetuneado el modelo y tenerlo en una carpeta, ej: "chatapp/fine_tuned_model"
            ft_model_path = "chatapp/fine_tuned_model"
            if ft_model is None:
                if not os.path.exists(ft_model_path):
                    return Response({"error": "El modelo fine-tuneado no existe."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                # Cargar modelo fine-tuneado
                from transformers import AutoTokenizer, AutoModelForCausalLM
                ft_tokenizer = AutoTokenizer.from_pretrained(ft_model_path)
                ft_instance = AutoModelForCausalLM.from_pretrained(ft_model_path, torch_dtype=torch.float16).to("cuda")
                ft_model = (ft_tokenizer, ft_instance)

            # Generar respuesta sin RAG, solo fine-tuneado (simple prompt)
            prompt = f"El usuario pregunta: '{user_query}'\nPor favor, responde con recomendaciones de películas."
            tokenizer, model = ft_model
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            output = model.generate(
                **inputs,
                max_length=500,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
            answer = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        elif mode == "hybrid":
            # Modo híbrido: si no existe hb_model, lo creamos
            ft_model_path = "chatapp/fine_tuned_model"
            if hb_model is None:
                if not os.path.exists(ft_model_path):
                    return Response({"error": "El modelo fine-tuneado no existe para modo híbrido."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                hb_model = HybridChatbot(rag_model=rag_model, finetuned_model_path=ft_model_path)

            answer = hb_model.generate_answer(user_query)
        else:
            return Response({"error": "Modo no válido"}, status=status.HTTP_400_BAD_REQUEST)

        return Response({"query": user_query, "answer": answer}, status=status.HTTP_200_OK)


class ResultsView(APIView):
    def get(self, request, format=None):
        # Ejemplo de resultados (puedes modificar a tu gusto)
        results = {
            "comparisons": [
                {"input": "Quiero ver una comedia romántica navideña", 
                 "RAG": "Te recomiendo 'Love Actually' ...", 
                 "FineTune": "Prueba 'The Holiday' ...", 
                 "Hybrid": "Basado en lo encontrado, quizás 'Love Actually' ..."}
            ],
            "metrics": {
                "BLEU": {"RAG": 0.25, "FineTune": 0.23, "Hybrid": 0.27},
                "CostEstimate": {"RAG": "Low", "FineTune": "Medium", "Hybrid": "Medium-Low"}
            }
        }
        return Response(results, status=status.HTTP_200_OK)
