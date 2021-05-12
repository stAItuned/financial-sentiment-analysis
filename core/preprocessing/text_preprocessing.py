from nltk.corpus import stopwords
from nltk import word_tokenize
import json
import nltk
import string
import re

from scripts.loadings import load_contractions_dict

porter = nltk.PorterStemmer()
lemmatizer = nltk.WordNetLemmatizer()

contractions_dictionary = load_contractions_dict()


def init_nltk():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')


def remove_punctuations(text):
    # download_stopwords()
    table = str.maketrans(dict.fromkeys(string.punctuation))

    return text.translate(table).strip()


def remove_stopwords(sentence, lang='english'):
    """
    :param sentence:        str
    :param lang:            str ['english' | 'italian' ]

    :return new_sentence:   str  --> sentence without stopwords
    """

    stop_words = stopwords.words(lang)

    new_sentence = ''
    for word in sentence.split(sep=' '):
        if word not in stop_words:
            new_sentence += f'{word} '

    return new_sentence

def remove_html(sentence):
    """
    :param sentence:        str

    :return new_sentence:   str  --> sentence without html tags
    """

    # . : any character
    # *? : 0 or more, ungreedy
    new_sentence = re.sub("<.*?>", '', sentence).strip()

    return new_sentence

def remove_links(sentence):
    """
    :param sentence:        str

    :return new_sentence:   str  --> sentence without links
    """

    # \S : not withe space
    # + : 1 or more
    # |-- it will cover both http and https
    new_sentence = re.sub("http\S+", '', sentence).strip()

    return new_sentence

def normalize_punctuation(sentence):
    """
    :param sentence:        str

    :return new_sentence:   str  --> sentence with a normalized punctuation
                                     e.g. !!!! -> !
                                          ??!! -> ?!
                                          ..., .., ...., -> ...
    """
    # extract each group of the same punctuation and maintain just the first match
    new_sentence = re.sub(r'([!?,;])\1+', r'\1', sentence)
    # each combination of dots (with two or more) will be normalized to three dots
    new_sentence = re.sub(r'\.{2,}', r'...', new_sentence)

    return new_sentence

def normalize_whitespaces(sentence):
    """
    :param sentence:        str

    :return new_sentence:   str  --> sentence with normalized spaces
    """

    # normalize one or more "spaces" into just a single space
    new_sentence = re.sub(r"( )\1+", r"\1", sentence)
    # normalize one or more "new lines" to just a single new line
    new_sentence = re.sub(r"(\n)\1+", r"\1", new_sentence)
    # normalize one or more "carriage returns" to just a single carriage return
    new_sentence = re.sub(r"(\r)\1+", r"\1", new_sentence)
    # normalize one or more "tabs" to just a single tab
    new_sentence = re.sub(r"(\t)\1+", r"\1", new_sentence)

    return new_sentence

def normalize_contractions(sentence,lang='english'):
    """
    :param sentence:        str
    :param lang:            str ['english' | 'italian' ]
                                 --> not implemented yet for italian

    :return new_sentence:   str  --> sentence with normalized contractions
                                     we use a dictionary provided by Wikipedia
    """

    if lang == "english":
        tokens = word_tokenize(sentence)

        for i, token in enumerate(tokens):
            if token in contractions_dictionary:
                tokens[i] = contractions_dictionary[token]

        new_sentence = ' '.join(tokens)
        return new_sentence

    # to do : manage italian contractions
    elif lang == "italian":
        return sentence


def normalize_char_sequences(sentence):
    """
    :param sentence:        str

    :return new_sentence:   str  --> sentence with normalized char sequences
                                     dropped to the second repetition
                                     e.g. Helllloooo -> Helloo
                                          Byeeee -> Byee
                                          Yeeees -> Yees
    """

    # take all the characters repeated from 2 to "n" times
    # replace them just with the first two chars
    new_sentence = re.sub(r'(.)\1{2,}', r'\1\1', sentence)

    return new_sentence

def clean_twitter(sentence):
    """
    :param sentence:        str

    :return:                str --> sentence without #, mentions and RT
    """

    # remove retweets
    # e.g. "RT @first_user"
    new_sentence = re.sub(r'RT( )@[A-z0-9]{1,}', '', sentence).strip()

    # remove "#" -> the text associated to the hashtag may contain useful informantions
    # remove non alphabetical chars
    new_sentence = re.sub("[^a-zA-Z]", " ", new_sentence).strip()

    return new_sentence


def stem_word(word):
    return porter.stem(word)


def stem_sentence(sentence):
    new_sentence = ''
    for word in sentence.split(sep=' '):
        new_sentence += f'{stem_word(word)} '

    return new_sentence

def lemmatize_word(word):
    ADJ, ADJ_SAT, ADV, NOUN, VERB = "a", "s", "r", "n", "v"
    lemmatized_word = f'{word}'

    for x in [ADJ, ADJ_SAT, ADV, NOUN, VERB]:
        lemmatized_word = lemmatizer.lemmatize(lemmatized_word, pos=x)

    return lemmatized_word


def lemmatize_sentence(sentence):
    new_sentence = ''
    for word in sentence.split(sep=' '):
        new_sentence += f'{lemmatize_word(word)} '

    return new_sentence


def sentence_splitter_by_nltk(text):
    return nltk.extract_test_sentences(text)
