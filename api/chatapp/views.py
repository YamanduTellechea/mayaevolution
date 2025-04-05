from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .rag_pipeline import RAGChatbot
from .gpt4_chatbot import GPT4Chatbot  # Importamos la clase GPT4Chatbot
from .models import ChatHistory
import os
import time

import traceback
import logging
from .serializers import ChatHistorySerializer
logger = logging.getLogger(__name__)

# Variables globales para almacenar modelos cargados
rag_model = None
gpt4_model = None

def initialize_rag_model():
    """
    Inicializa el modelo RAG si aún no está cargado.
    """
    global rag_model
    if rag_model is None:
        print("Inicializando modelo RAG...")
        dataset_path = "chatapp/dataset"
        embedding_file = "movie_embeddings.npz"
        rag_model = RAGChatbot(
            dataset_path=dataset_path            
        )
        print("Modelo RAG inicializado correctamente.")

def initialize_gpt4_model():
    """
    Inicializa el modelo GPT-4 si aún no está cargado.
    """
    global gpt4_model
    if gpt4_model is None:
        print("Inicializando modelo GPT-4...")
        gpt4_model = GPT4Chatbot()
        print("Modelo GPT-4 inicializado correctamente.")

class ChatView(APIView):
    def post(self, request, format=None):
        """
        Maneja las solicitudes POST para generar respuestas según el modo seleccionado.
        """
        user_query = request.data.get("query", "")
        mode = request.data.get("mode", "rag")  # 'rag', 'gpt4'

        logger.debug(f"Solicitud recibida. Modo: {mode}, Consulta: {user_query}")        

        try:
            start_time = time.time()
            if mode == "rag":
                initialize_rag_model()  # Inicializar RAG si es necesario
                answer = rag_model.generate_answer(user_query)

            elif mode == "gpt4":                
                initialize_gpt4_model()  # Inicializar GPT-4 si es necesario
                answer = gpt4_model.generate_answer(user_query)

            else:
                return Response({"error": "Modo no válido"}, status=status.HTTP_400_BAD_REQUEST)
            
            end_time = time.time()  # Finalizar medición de tiempo
            response_time = round(end_time - start_time, 3)  # Convertir a milisegundos

            # Estimación del coste de GPT-4
            estimated_cost = 0.0
            if mode == "gpt4":
                estimated_cost = 0.06  # Ajustar según API de OpenAI

            interaction = ChatHistory.objects.create(
                query=user_query,
                mode=mode,
                response=answer,
                response_time=response_time,
                cost=estimated_cost
            )
            logger.info(f"Interacción registrada: {interaction}")

            # Devolver la respuesta junto con métricas
            return Response({
                "query": user_query,
                "answer": answer,
                "time": response_time,
                "cost": estimated_cost
            }, status=status.HTTP_200_OK)
       
        except Exception as e:
            print("Error interno en ChatView:")
            traceback.print_exc()  # Esto imprime el error completo en consola
            return Response({"error": f"Error al procesar la solicitud: {str(e)}"},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ChatHistoryView(APIView):
    def get(self, request, *args, **kwargs):
        """
        Maneja las solicitudes GET para devolver el historial de interacciones.
        """
        try:
            # Recupera todos los registros de la base de datos ordenados por fecha
            history = ChatHistory.objects.all().order_by("-timestamp")
            serializer = ChatHistorySerializer(history, many=True)  # Serializa los datos
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            # Si hay un error, devuélvelo con un mensaje apropiado
            return Response(
                {"error": f"Error al obtener el historial: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
class ResultsView(APIView):
    def get(self, request, format=None):
        """
        Devuelve un ejemplo de resultados para fines de demostración.
        """
        # Ejemplo de datos de comparación
        results = {
            "comparisons": [
                {"input": "Quiero ver una comedia romántica navideña",
                 "RAG": "Te recomiendo 'Love Actually' ...",
                 "GPT-4": "Prueba 'The Holiday', es una opción perfecta para una noche navideña romántica."}
            ],
            "metrics": {
                "BLEU": {"RAG": 0.25, "GPT-4": 0.35},
                "CostEstimate": {"RAG": "Low", "GPT-4": "High"}
            }
        }
        return Response(results, status=status.HTTP_200_OK)