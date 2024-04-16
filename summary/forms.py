from django import forms
from .models import UploadedFile



class PdfUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedFile
        fields = ["pdf_file"]
