from nltk.corpus import stopwords
import nltk
import string

porter = nltk.PorterStemmer()
lemmatizer = nltk.WordNetLemmatizer()


def init_nltk():
    nltk.download('stopwords')
    nltk.download('wordnet')


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
