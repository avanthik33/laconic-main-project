from django.urls import path
from .views import signup,signin,logout_view

urlpatterns = [
    path("signup/", signup, name="signup"),
    path("", signin, name="signin"),
    path("logout/", logout_view, name="logout"),
    # path("profile/", profile, name="profile"),
]
