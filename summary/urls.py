from django.urls import path
from .views import summarize, summarize_pdf,about

urlpatterns = [
    path("summarize/", summarize, name="summarize"),
    path("summarize_pdf/", summarize_pdf, name="summarize_pdf"),
    path("about/", about, name="about"),
]
