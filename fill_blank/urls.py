from django.urls import path
from .views import generate_questions

urlpatterns = [
    path("generate/", generate_questions, name="generate_questions"),
    # Add other URL patterns as needed
]
