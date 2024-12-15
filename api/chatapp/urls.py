from django.urls import path
from .views import ChatView, ResultsView

urlpatterns = [
    path('chat/', ChatView.as_view(), name='chat'),
    path('results/', ResultsView.as_view(), name='results'),
]
