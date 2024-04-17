from django.urls import path
from .views import index, generate_mcq

urlpatterns = [
    path("generate/", index, name="index"),
    path("mcq/", generate_mcq, name="generate_mcq"),
    # Add other URL patterns as needed
]
