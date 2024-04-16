from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login , logout
from django.contrib.auth.hashers import make_password
from .models import publicUser


def signup(request):
    if request.method == "POST":
        username = request.POST["userName"]
        email = request.POST["email"]
        password = request.POST["password"]
        confirm_password = request.POST["confirm_password"]

        # Check if passwords match
        if password != confirm_password:
            return render(
                request,
                "authen/signup.html",
                {"error_message": "Passwords do not match"},
            )

        if publicUser.objects.filter(userName=username).exists():
            return render(
                request,
                "authen/signup.html",
                {"error_message": "Username already exists"},
            )
        elif publicUser.objects.filter(email=email).exists():
            return render(
                request, "authen/signup.html", {"error_message": "Email already exists"}
            )

        # Hash the password

        # Create the new user
        new_user = publicUser(userName=username, email=email, password=password)
        new_user.save()
        return redirect("signin")
    else:
        return render(request, "authen/signup.html")


def signin(request):
    if request.method == "POST":
        username = request.POST["userName"]
        password = request.POST["password"]

        # Check if user exists
        try:
            user = publicUser.objects.get(userName=username)
        except publicUser.DoesNotExist:
            return render(
                request,
                "authen/signin.html",
                {"error_message": "Invalid username or password"},
            )

        # Check password
        if user.password != password:
            return render(
                request,
                "authen/signin.html",
                {"error_message": "Invalid username or password"},
            )

        return redirect("summarize_pdf")  # Redirect to home page after successful login
    else:
        return render(request, "authen/signin.html")


def logout_view(request):
    logout(request)
    return redirect("signin")

from django.contrib.auth.decorators import login_required
from .models import publicUser

# def profile(request):
#     user = request.user  # Retrieve the signed-in user
#     return render(request, "authen/profile.html", {"user": user})
