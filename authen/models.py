# models.py

from django.contrib.auth.hashers import make_password, check_password
from django.db import models


class publicUser(models.Model):
    userName = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=128)
