from django.shortcuts import render
import spacy


# def generate_questions(request):
#     if request.method == "POST":
#         paragraph = request.POST.get("paragraph", "")
#         questions = generate_questions(paragraph)
#         return render(request, "fill_blank/questions.html", {"questions": questions})
#     else:
#         return render(request, "fill_blank/index.html")


from django.shortcuts import render
from transformers import T5ForConditionalGeneration, T5Tokenizer
from django.shortcuts import render
from transformers import BartForConditionalGeneration, BartTokenizer


def generate_questions(request):
    if request.method == "POST":
        paragraph = request.POST.get("paragraph", "")

        # Load pre-trained BART model and tokenizer
        model_name = "facebook/bart-large-cnn"
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)

        # Tokenize paragraph
        inputs = tokenizer(
            [paragraph], return_tensors="pt", max_length=1024, truncation=True
        )

        # Generate questions
        output_ids = model.generate(
            inputs.input_ids, attention_mask=inputs.attention_mask, max_length=64
        )

        # Decode generated questions
        generated_questions = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )

        # Render the HTML page with generated questions
        return render(
            request,
            "fill_blank/questions.html",
            {"paragraph": paragraph, "questions": generated_questions},
        )
    else:
        return render(request, "fill_blank/index.html")
