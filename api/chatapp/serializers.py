from rest_framework import serializers
from .models import ChatHistory

class ChatHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatHistory
        fields = ["query", "mode", "response", "response_time", "cost", "timestamp"]
