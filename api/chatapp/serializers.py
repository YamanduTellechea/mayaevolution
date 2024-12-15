from rest_framework import serializers

class ChatSerializer(serializers.Serializer):
    query = serializers.CharField()
    mode = serializers.CharField()
