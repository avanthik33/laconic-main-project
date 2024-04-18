from django.urls import path
from .views import index

urlpatterns = [
    path("generate/", index, name="index"),
    # path('mcq/', mcq_index, name='mcq_index'),
    # Add other URL patterns as needed
]
