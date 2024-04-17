from django.shortcuts import render
import spacy


# def generate_questions(request):
#     if request.method == "POST":
#         paragraph = request.POST.get("paragraph", "")
#         questions = generate_questions(paragraph)
#         return render(request, "fill_blank/questions.html", {"questions": questions})
#     else:
#         return render(request, "fill_blank/index.html")

import re

from django.shortcuts import render
import re
import random
import spacy

# Load the English language model for spaCy
nlp = spacy.load("en_core_web_sm")


def generate_fill_in_the_blank(text, num_blanks=1):
    # Split the text into sentences
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)

    # Initialize lists to store fill-in-the-blank questions and options for blanked words
    fill_in_the_blank_questions = []
    options_for_blanks = []

    for sentence in sentences:
        # Tokenize the sentence using spaCy
        doc = nlp(sentence)

        # Filter out stop words, punctuation, and entities
        filtered_words = [token.text for token in doc if not token.is_stop and not token.is_punct and not token.ent_type]

        # If there are enough words in the sentence, create a fill-in-the-blank question
        if len(filtered_words) >= num_blanks:
            # Randomly shuffle the filtered words
            random.shuffle(filtered_words)

            # Select the first `num_blanks` words
            selected_words = filtered_words[:num_blanks]

            # Create a fill-in-the-blank question with blanks
            fill_in_the_blank_question = sentence
            for word in selected_words:
                fill_in_the_blank_question = fill_in_the_blank_question.replace(word, "_____")

            fill_in_the_blank_questions.append(fill_in_the_blank_question.strip())
            options_for_blanks.append(selected_words)

    return fill_in_the_blank_questions, options_for_blanks


def generate_mcq(text, num_questions=5, num_options=4):
    # Split the text into sentences
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)

    # Initialize lists to store MCQ questions and options
    mcq_questions = []

    for _ in range(num_questions):
        # Randomly select a sentence
        sentence = random.choice(sentences)

        # Tokenize the sentence using spaCy
        doc = nlp(sentence)

        # Filter out stop words, punctuation, and entities
        filtered_words = [
            token.text
            for token in doc
            if not token.is_stop and not token.is_punct and not token.ent_type
        ]

        # If there are enough words in the sentence, create a MCQ question
        if len(filtered_words) >= num_options:
            # Randomly shuffle the filtered words
            random.shuffle(filtered_words)

            # Select the first `num_options` words as options
            options = filtered_words[:num_options]

            # Randomly select the correct option
            correct_option = random.choice(options)

            # Create the MCQ question
            question = f"What is the meaning of '{correct_option}' in the following sentence? '{sentence}'"
            mcq_options = [(option, option == correct_option) for option in options]

            mcq_questions.append((question, mcq_options))

    return mcq_questions


def index(request):
    paragraph = ""
    questions = []
    if request.method == "POST":
        paragraph = request.POST.get("paragraph", "")
        questions = generate_fill_in_the_blank(paragraph, num_blanks=1)
        mcq=generate_mcq(paragraph)
    return render(
        request,
        "fill_blank/index.html",
        {"paragraph": paragraph, "questions": questions,"mcq":mcq},
    )
