from django import forms
from .models import Paragraph


class ParagraphForm(forms.ModelForm):
    class Meta:
        model = Paragraph
        fields = ["content"]
