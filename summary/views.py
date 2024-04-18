from django.shortcuts import render
from django.http import JsonResponse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .forms import PdfUploadForm
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from django.views.decorators.csrf import csrf_exempt
from django.core.exceptions import ValidationError
import PyPDF2
from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
import os
from sklearn.feature_extraction.text import CountVectorizer
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
import nltk


def about(request):
    return render(request, "about.html")



import matplotlib.pyplot as plt
from rouge import Rouge
import io
import base64
import matplotlib
matplotlib.use("Agg")


def summarize(request):
    if request.method == "POST":
        paragraph = request.POST.get("paragraph", "")
        max_summary_length = int(request.POST.get("max_summary_length", 100))
        word_count = len(paragraph.split())  # Count words in the paragraph
        summary = summary_function(paragraph, max_summary_length)
        simple_summary = simple_summary_function(paragraph, max_summary_length)
        general_summary = general_summary_function(paragraph, max_summary_length)

        # Evaluation using Rouge
        rouge = Rouge()
        rouge_scores_summary = rouge.get_scores(summary, paragraph)[0]
        rouge_scores_simple_summary = rouge.get_scores(simple_summary, paragraph)[0]
        rouge_scores_general_summary = rouge.get_scores(general_summary, paragraph)[0]

        # Calculate average F1 scores
        avg_rouge_f1_summary = (
            rouge_scores_summary["rouge-1"]["f"]
            + rouge_scores_summary["rouge-2"]["f"]
            + rouge_scores_summary["rouge-l"]["f"]
        ) / 3
        avg_rouge_f1_simple_summary = (
            rouge_scores_simple_summary["rouge-1"]["f"]
            + rouge_scores_simple_summary["rouge-2"]["f"]
            + rouge_scores_simple_summary["rouge-l"]["f"]
        ) / 3
        avg_rouge_f1_general_summary = (
            rouge_scores_general_summary["rouge-1"]["f"]
            + rouge_scores_general_summary["rouge-2"]["f"]
            + rouge_scores_general_summary["rouge-l"]["f"]
        ) / 3

        # Plotting
        labels = ["Summary 3", "Summary 2", "Summary 1"]
        scores = [
            avg_rouge_f1_simple_summary,
            avg_rouge_f1_general_summary,
            avg_rouge_f1_summary,
        ]

        plt.bar(labels, scores)
        plt.xlabel("Summary Type")
        plt.ylabel("Quality")
        plt.title("Overall Quality of Summaries")
        plt.ylim(0, 1)  # Set y-axis limit between 0 and 1

        # Convert plot to image buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        # Encode the image buffer to base64
        plot_data = base64.b64encode(buf.read()).decode("utf-8")

        return JsonResponse(
            {
                "summary": summary,
                "general_summary": general_summary,
                "extractive_summary": simple_summary,
                "word_count": word_count,
                "avg_rouge_f1_summary": avg_rouge_f1_summary,
                "avg_rouge_f1_simple_summary": avg_rouge_f1_simple_summary,
                "avg_rouge_f1_general_summary": avg_rouge_f1_general_summary,
                "plot_data": plot_data,  # Pass the plot as base64 encoded string
            }
        )
    return render(request, "text/summarize.html")


###########################################################################
# summary by using textRank algorithm                                  1


from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import spacy


def general_summary_function(text, max_summary_length):
    # Create a parser using the plaintext input
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    # Get the summary with the specified number of sentences
    summary = summarizer(parser.document, max_summary_length)
    # Enhance summary with named entity recognition
    nlp = spacy.load("en_core_web_sm")
    named_entities = set()
    for sentence in summary:
        doc = nlp(str(sentence))
        for ent in doc.ents:
            named_entities.add(ent.text)
    # Filter sentences containing named entities
    filtered_summary = [
        str(sentence)
        for sentence in summary
        if any(entity in str(sentence) for entity in named_entities)
    ]
    return " ".join(filtered_summary)


###############################################################################
# text summarization algorithm using the PageRank algorithm                2


import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords


def summary_function(text, max_summary_length):
    # Tokenize sentences
    sentences = sent_tokenize(text)
    # Remove stop words
    stop_words = stopwords.words("english")
    # Calculate TF-IDF (Term Frequency-Inverse Document Frequency)
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    # Calculate cosine similarity between sentences
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
    # Build sentence similarity graph
    graph = {}
    for i in range(len(sentences)):
        graph[i] = []
        for j in range(len(sentences)):
            if (
                i != j and cosine_similarities[i][j] > 0.2
            ):  # Adjust the threshold as needed
                graph[i].append(j)
    # Rank sentences using PageRank algorithm
    scores = {i: 1 for i in range(len(sentences))}
    d = 0.85  # Damping factor
    for _ in range(10):  # Iterations for convergence
        for i in range(len(sentences)):
            score = 1 - d
            for j in graph:
                if i in graph[j]:
                    score += d * (1 / len(graph[j])) * scores[j]
            scores[i] = score
    # Sort sentences by score and select top sentences as summary
    ranked_sentences = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)), reverse=True
    )
    # Enhance summary with sentence position and named entity recognition
    enhanced_summary = []
    nlp = spacy.load("en_core_web_sm")
    named_entities = set()
    for score, sentence in ranked_sentences:
        doc = nlp(sentence)
        for ent in doc.ents:
            named_entities.add(ent.text)
        enhanced_summary.append((score, sentence))
    # Filter sentences containing named entities
    filtered_summary = [
        sentence
        for score, sentence in enhanced_summary
        if any(entity in sentence for entity in named_entities)
    ]
    # Choose the number of sentences in the summary
    summary_length = min(max_summary_length, len(filtered_summary))
    # Combine top sentences to form the summary
    summary = " ".join(filtered_summary[:summary_length])
    return summary


###############################################################################
# using count vectorizer                                                 3


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import re
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def simple_summary_function(text, max_summary_length):
    # Preprocess the text
    text = re.sub(r"\s+", " ", text)

    # Tokenize sentences
    sentences = re.split("(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)

    # Extract features from the text
    vectorizer = CountVectorizer().fit(sentences)
    vectors = vectorizer.transform(sentences)

    # Calculate cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(vectors, vectors)

    # Rank sentences based on cosine similarity scores
    sentence_scores = cosine_sim_matrix.sum(axis=1)

    # Enhance summary with sentence position and named entity recognition
    enhanced_summary = []
    nlp = spacy.load("en_core_web_sm")
    named_entities = set()
    for idx, sentence in enumerate(sentences):
        doc = nlp(sentence)
        for ent in doc.ents:
            named_entities.add(ent.text)
        enhanced_summary.append((sentence_scores[idx], sentence))

    # Filter sentences containing named entities
    filtered_summary = [
        sentence
        for score, sentence in enhanced_summary
        if any(entity in sentence for entity in named_entities)
    ]

    # Choose the number of sentences in the summary
    summary_length = min(max_summary_length, len(filtered_summary))

    # Combine top sentences to form the summary
    summary = " ".join(filtered_summary[:summary_length])

    return summary


#####################################################################################
#                                                                          PDF

import json

import matplotlib.pyplot as plt
from rouge import Rouge
import io
import base64


def summarize_pdf(request):
    if request.method == "POST":
        uploaded_file = request.FILES["file"]
        if uploaded_file.name.endswith(".pdf"):
            # Extract text from PDF
            text = extract_text_from_pdf(uploaded_file)
            max_summary_length_pdf = int(
                request.POST.get("max_summary_length_pdf", 100)
            )
            word_count = len(text.split())

            summary = summary_function(text, max_summary_length_pdf)
            simple_summary = simple_summary_function(text, max_summary_length_pdf)
            general_summary = general_summary_function(text, max_summary_length_pdf)

            # Evaluation using Rouge
            rouge = Rouge()
            rouge_scores_summary = rouge.get_scores(summary, text)[0]
            rouge_scores_simple_summary = rouge.get_scores(simple_summary, text)[0]
            rouge_scores_general_summary = rouge.get_scores(general_summary, text)[0]

            # Calculate average F1 scores
            avg_rouge_f1_summary = (
                rouge_scores_summary["rouge-1"]["f"]
                + rouge_scores_summary["rouge-2"]["f"]
                + rouge_scores_summary["rouge-l"]["f"]
            ) / 3
            avg_rouge_f1_simple_summary = (
                rouge_scores_simple_summary["rouge-1"]["f"]
                + rouge_scores_simple_summary["rouge-2"]["f"]
                + rouge_scores_simple_summary["rouge-l"]["f"]
            ) / 3
            avg_rouge_f1_general_summary = (
                rouge_scores_general_summary["rouge-1"]["f"]
                + rouge_scores_general_summary["rouge-2"]["f"]
                + rouge_scores_general_summary["rouge-l"]["f"]
            ) / 3

            # Plotting
            labels = ["Summary 3", "Summary 1", "Summary 2"]
            scores = [
                avg_rouge_f1_simple_summary,
                avg_rouge_f1_general_summary,
                avg_rouge_f1_summary,
            ]

            # Convert plot to base64 encoded string
            plt.bar(labels, scores)
            plt.xlabel("Summary Type")
            plt.ylabel("Quality")
            plt.title("Overall Quality of Summaries")
            plt.ylim(0, 1)  # Set y-axis limit between 0 and 1

            # Convert plot to image buffer
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plt.close()

            # Encode the image buffer to base64
            plot_data = base64.b64encode(buf.read()).decode("utf-8")

            return JsonResponse(
                {
                    "general_summary": general_summary,
                    "summary": summary,
                    "extractive_summary": simple_summary,
                    "word_count": word_count,
                    "avg_rouge_f1_summary": avg_rouge_f1_summary,
                    "avg_rouge_f1_simple_summary": avg_rouge_f1_simple_summary,
                    "avg_rouge_f1_general_summary": avg_rouge_f1_general_summary,
                    "plot_data": plot_data,  # Pass the plot as base64 encoded string
                }
            )
        else:
            return JsonResponse({"error": "Uploaded file is not a PDF"})

    return render(request, "pdf/summarize_pdf.html")


def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text


# from transformers import pipeline


# def bert_summarize(text, max_len):
#     # Load the BERT-based summarization pipeline
#     summarizer = pipeline("summarization")

#     # Summarize the text using BERT
#     summarized_text = summarizer(
#         text, max_length=max_len, min_length=max_len // 2, do_sample=False
#     )[0]["summary_text"]

#     return summarized_text


#####################################################################################
# pdf1


# nltk.download("stopwords")


# @csrf_exempt
# def upload_pdf(request):
#     if request.method == "POST":
#         form = PdfUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             uploaded_file = form.save()

#             # Extract text from PDF
#             pdf_text = extract_text_from_pdf(uploaded_file.pdf_file.path)

#             # Get summary length from request
#             summary_length = int(request.POST.get("summary_length", 3))

#             # Summarize text
#             summary = get_summary(pdf_text, summary_length)

#             # Delete the uploaded file
#             uploaded_file.delete()

#             return JsonResponse({"summary": summary})

#     else:
#         form = PdfUploadForm()

#     return render(request, "summarizer/upload_pdf.html", {"form": form})


# @csrf_exempt
# def update_summary_length(request):
#     if request.method == "POST":
#         summary_length = int(request.POST.get("summary_length", 3))
#         # You can store the summary length in session or a database for future use
#         # For simplicity, I'm just returning the updated length as JSON response
#         return JsonResponse({"summary_length": summary_length})


# def extract_text_from_pdf(file_path):
#     text = ""
#     with open(file_path, "rb") as f:
#         reader = PdfReader(f)
#         for page in reader.pages:
#             text += page.extract_text()
#     return text


# def get_summary(text, summary_length):
#     # Tokenize the text into sentences
#     sentences = sent_tokenize(text)

#     # Remove stopwords
#     stop_words = set(stopwords.words("english"))
#     clean_sentences = [
#         sentence for sentence in sentences if sentence.lower() not in stop_words
#     ]

#     # Calculate TF-IDF scores
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform(clean_sentences)

#     # Calculate pairwise cosine similarity between sentences
#     similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

#     # Rank sentences based on cosine similarity
#     sentence_scores = [(i, sum(similarity_matrix[i])) for i in range(len(sentences))]
#     sentence_scores = sorted(sentence_scores, key=lambda x: x[1], reverse=True)

#     # Select top sentences as summary
#     summary_length = min(summary_length, len(sentence_scores))
#     summary_sentences = [
#         sentences[score[0]] for score in sentence_scores[:summary_length]
#     ]
#     summary = " ".join(summary_sentences)

#     return summary


# from django.shortcuts import render
# import spacy
# import random

# from sklearn.feature_extraction.text import TfidfVectorizer
# import random


# def generate_fill_in_the_blank(paragraph):
#     # Load English tokenizer, tagger, parser, NER, and word vectors
#     nlp = spacy.load("en_core_web_sm")

#     # Process the paragraph with spaCy
#     doc = nlp(paragraph)

#     # Get the words from the document
#     words = [token.text for token in doc]

#     # Convert the paragraph into a single string for TF-IDF calculation
#     paragraph_text = " ".join(words)

#     # Calculate TF-IDF scores for each word
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform([paragraph_text])

#     # Get feature names (words) and corresponding TF-IDF scores
#     feature_names = vectorizer.get_feature_names_out()
#     tfidf_scores = tfidf_matrix.toarray()[0]

#     # Combine words with their TF-IDF scores into a dictionary
#     word_tfidf = dict(zip(feature_names, tfidf_scores))

#     # Sort words by their TF-IDF scores in descending order
#     sorted_words = sorted(word_tfidf.items(), key=lambda x: x[1], reverse=True)

#     # Keep track of the number of blanks added
#     num_blanks = 0

#     # Create a list to store the fill-in-the-blank questions
#     fill_in_paragraph = []
#     options = set()  # Use a set to ensure uniqueness of options
#     blanked_words = set()  # Keep track of words that have been blanked

#     # Define the maximum number of blanks to generate
#     max_blanks = 10

#     # Iterate over each token in the document
#     for token in doc:
#         # Check if the token is among the most important words and it's a noun, verb, adjective, or adverb
#         if token.text in [
#             word[0] for word in sorted_words[:max_blanks]
#         ] and token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]:
#             # Replace the word with a blank placeholder if it's not already blanked
#             if token.text not in blanked_words and num_blanks < max_blanks:
#                 fill_in_paragraph.append("_____")
#                 num_blanks += 1

#                 # Add the word itself to options
#                 options.add(token.text)

#                 # Get related words for the word
#                 related_words = [
#                     child.text
#                     for child in token.children
#                     if child.pos_ in ["ADJ", "ADV", "NOUN", "VERB"]
#                 ]
#                 options.update(related_words)  # Add related words to options

#                 # Mark the word as blanked
#                 blanked_words.add(token.text)
#             else:
#                 fill_in_paragraph.append(token.text)
#         else:
#             # Keep the token as it is if it's not a word of interest
#             fill_in_paragraph.append(token.text)

#     # Shuffle the options list
#     options = list(options)
#     random.shuffle(options)

#     # Join the tokens back into a string
#     fill_in_paragraph = " ".join(fill_in_paragraph)

#     return fill_in_paragraph, options[:max_blanks]


# def fill_in_the_blank(request):
#     fill_in_paragraph = None
#     options = None

#     if request.method == "POST":
#         paragraph = request.POST.get("paragraph", "")
#         # Generate fill-in-the-blank questions and options
#         fill_in_paragraph, options = generate_fill_in_the_blank(paragraph)

#     return render(
#         request,
#         "fill_blank/fill_blank.html",
#         {"fill_in_paragraph": fill_in_paragraph, "options": options},
#     )
