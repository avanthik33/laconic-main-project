import spacy
import random
from nltk.corpus import wordnet
import nltk
from nltk.corpus import stopwords

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")

# Load English language model
nlp = spacy.load("en_core_web_sm")

# Define the set of stopwords
STOP_WORDS = set(stopwords.words("english"))


def generate_questions_for_paragraph(paragraph):
    """
    Generate meaningful questions for each sentence in a paragraph.

    Args:
    - paragraph (str): The input paragraph.

    Returns:
    - List of dictionaries, each containing a question and its options.
    """
    doc = nlp(paragraph)
    sentences = [sent.text for sent in doc.sents]

    questions = []

    for sentence in sentences:
        # Create a question for each sentence in the paragraph
        question_text = (
            f"What is the meaning of the following sentence?\n\n'{sentence}'"
        )
        options = generate_options(sentence)
        questions.append({"question": question_text, "options": options})

    return questions

from nltk.corpus import wordnet
import random


def generate_options(sentence):
    options = []

    # Tokenize the sentence and get related words for each token
    for token in nlp(sentence):
        # Add the original token as an option
        options.append(token.text)

        # Get synonyms and hypernyms of the token from WordNet
        synonyms = set()
        hypernyms = set()
        for syn in wordnet.synsets(token.text):
            # Check if the synonym is relevant to the context
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ")
                # Filter out synonyms that are stopwords or punctuation
                if (
                    synonym.lower() not in STOP_WORDS
                    and not synonym.isdigit()
                    and synonym != token.text
                ):
                    synonyms.add(synonym)
            # Check if the hypernym is relevant to the context
            for hypernym in syn.hypernyms():
                for lemma in hypernym.lemmas():
                    hypernym_word = lemma.name().replace("_", " ")
                    # Filter out hypernyms that are stopwords or punctuation
                    if (
                        hypernym_word.lower() not in STOP_WORDS
                        and not hypernym_word.isdigit()
                        and hypernym_word != token.text
                    ):
                        hypernyms.add(hypernym_word)

        # Add synonyms and hypernyms to the options
        options.extend(list(synonyms) + list(hypernyms))

    # Remove duplicates and shuffle the options
    options = list(set(options))
    random.shuffle(options)

    # Return a maximum of 4 options
    return options[:4]
