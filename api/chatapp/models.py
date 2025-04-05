from django.db import models

class ChatHistory(models.Model):
    query = models.TextField()  # Pregunta del usuario
    mode = models.CharField(max_length=10, choices=[("rag", "RAG"), ("gpt4", "GPT-4")])  # Modo de respuesta
    response = models.TextField()  # Respuesta generada
    response_time = models.FloatField()  # Tiempo de respuesta en segundos
    cost = models.FloatField(default=0.0)  # Coste (solo para GPT-4)
    timestamp = models.DateTimeField(auto_now_add=True)  # Fecha y hora del mensaje

    def __str__(self):
        return f"{self.timestamp} - {self.mode} - {self.query[:50]}"