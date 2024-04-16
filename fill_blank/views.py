from django.shortcuts import render, redirect
from .forms import ParagraphForm
from .utils import generate_questions_for_paragraph
from .models import Paragraph


def create_paragraph(request):
    if request.method == "POST":
        form = ParagraphForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("generate_questions")
    else:
        form = ParagraphForm()
    return render(request, "fill_blank/paragraph.html", {"form": form})


def generate_questions(request):
    paragraphs = Paragraph.objects.all()
    questions = []
    for paragraph in paragraphs:
        # Ensure that the paragraph content is passed to the generation function
        questions.extend(generate_questions_for_paragraph(paragraph.content))
    return render(request, "fill_blank/questions.html", {"questions": questions})
