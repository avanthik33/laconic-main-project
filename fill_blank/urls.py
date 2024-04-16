from django.urls import path
from .views import create_paragraph, generate_questions

urlpatterns = [
    path("create/", create_paragraph, name="create_paragraph"),
    path("generate/", generate_questions, name="generate_questions"),
    # Add other URL patterns as needed
]
