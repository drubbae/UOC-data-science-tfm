# https://towardsdatascience.com/nlp-building-text-cleanup-and-preprocessing-pipeline-eba4095245a0
from langdetect import detect
import pandas as pd
import nltk
nltk.download('wordnet')
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
import spacy
import re
import emoji
import string
import unicodedata
print('package --> pandas --> version --> ' + pd.__version__)
print('package --> re     --> version --> ' + re.__version__)
print('package --> ntlk   --> version --> ' + nltk.__version__)


debug = True
test = True
step = 0
path_input = r'D:\master\data science\semestre 4\M2.979 - tfm\data'
path_output = r'D:\master\data science\semestre 4\M2.979 - tfm\data'


# ----------- READ DATA
# load parsed data1
if test:
    df = pd.read_csv(path_input + r'\01_data_test.csv')
    if debug:
        df = df.filter(['content'])
else:
    df = pd.read_csv(path_input + r'\01_data.csv')
    if debug:
        df = df.filter(['content'])
# -----------------------------


# ---------- STEP
# ---------- DELETE COLON
# remove punctuation
re_pattern = r'(:)'
df['content'] = df['content'].apply(lambda x: re.sub(re_pattern, ' ', str(x)))
step += 1
step_name = '_delete_colon'
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
# -----------------------------


# ---------- STEP
# ---------- EMOJIS
# replace emoji by its short name between ':' --> e.g. --> :face_blowing_a_kiss:
df['content'] = df['content'].map(lambda x: emoji.demojize(x))
step += 1
step_name = '_replace_emojis'
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
# remove emoji's short name between ':'
re_pattern = r'\:(.*?)\:'
df['content'] = df['content'].apply(lambda x: re.sub(re_pattern, ' ', x))
step += 1
step_name = '_delete_emojis'
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
# -----------------------------


# ---------- STEP
# ---------- EMAILS
# extract emails, and save them into a new field named 'email'
re_pattern = r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'
df['email'] = df['content'].apply(lambda x: re.findall(re_pattern, x))
# remove emails
df['content'] = df['content'].apply(lambda x: re.sub(re_pattern, ' ', x))
step += 1
step_name = '_delete_emails'
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
# -----------------------------


# ---------- STEP
# ---------- WEBSITES
# extract websites, and save them into a new field named 'website'
re_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
df['website'] = df['content'].apply(lambda x: re.findall(re_pattern, x))
# remove websites
df['content'] = df['content'].apply(lambda x: re.sub(re_pattern, ' ', x))
step += 1
step_name = '_delete_websites'
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
# -----------------------------


# ---------- STEP
# ---------- HASHTAGS
# https://towardsdatascience.com/basic-tweet-preprocessing-in-python-efd8360d529e
# extract hashtags, and save them into a new field named 'hashtag'
re_pattern = r'#(\w+)'
df['hashtag'] = df['content'].apply(lambda x: re.findall(re_pattern, x))
# remove the hashtag identifier symbol
re_pattern = r'(#)'
df['content'] = df['content'].apply(lambda x: re.sub(re_pattern, ' ', x))
step += 1
step_name = '_delete_hashtags'
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
# -----------------------------


# ---------- STEP
# ---------- MENTIONS
# extract mentions, and save them into a new field named 'mention'
re_pattern = r'@(\w+)'
df['mention'] = df['content'].apply(lambda x: re.findall(re_pattern, x))
# remove the mention identifier symbol
re_pattern = r'(@\w+)'
df['content'] = df['content'].apply(lambda x: re.sub(re_pattern, ' ', x))
step += 1
step_name = '_delete_mentions'
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
# -----------------------------


# ---------- STEP
# ---------- ACCENTED
# https://towardsdatascience.com/nlp-building-text-cleanup-and-preprocessing-pipeline-eba4095245a0
# remove accented characters


# function to remove accented characters
def remove_accented_chars(text):
   new_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
   return new_text


# call function to remove accented characters (remove_accented_chars)
df['content'] = df['content'].map(lambda x: remove_accented_chars(x))
step += 1
step_name = '_delete_accented'
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
# -----------------------------


# ---------- STEP
# ---------- NUMBERS
# remove numbers
re_pattern = r'[^a-zA-z.,!?/:;\"\'\s]'
df['content'] = df['content'].apply(lambda x: re.sub(re_pattern, ' ', x))
step += 1
step_name = '_delete_numbers'
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
# -----------------------------


# ---------- STEP
# ---------- PUNCTUATION
# remove punctuation
df['content'] = df['content'].apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))
step += 1
step_name = '_delete_punctuation'
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
# -----------------------------


# ---------- STEP
# ---------- LOWERCASE
# lowercase the text
df['content'] = df['content'].map(lambda x: x.lower())
step += 1
step_name = '_lowercase'
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
# -----------------------------


# ---------- STEP
# ---------- WHITESPACES
# remove extra spaces
re_pattern = r' +'
df['content'] = df['content'].apply(lambda x: re.sub(re_pattern, ' ', x))
step += 1
step_name = '_whitespaces'
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
# -----------------------------


# ---------- STEP
# ---------- LANGUAGE
# https://medium.com/1001-nights-in-data-science/three-methods-of-detecting-language-of-textual-data-2c7cc41033b1
# function to detect language
def detect_language(text):
    try:
        language_detected = detect(text)
    except:
        language_detected = ''
    return language_detected


# call function to detect language (detect_language)
df['language'] = df['content'].map(lambda x: detect_language(x))
step += 1
step_name = '_language'
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
# -----------------------------


# ---------- STEP
# ---------- SELECT LANGUAGE (ENGLISH)
# select only data in english
df = df[df['language'] == 'en']
step += 1
step_name = '_select_english'
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
# -----------------------------


# ---------- STEP
# ---------- TOKENIZE
tknzr = TweetTokenizer()
df['content'] = df['content'].map(lambda x: tknzr.tokenize(x))
step += 1
step_name = '_tokenize'
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
# -----------------------------



# ---------- STEP
# ---------- WORDSEGMENT
# # https://medium.com/analytics-vidhya/working-with-twitter-data-b0aa5419532
from wordsegment import load, segment
load()


def get_wordsegment(text):
    text = [segment(word) for word in text]
    return text
def test(list_of_lists):
    result = [item for elem in list_of_lists for item in elem]
    return result

# call function
df['content'] = df['content'].map(lambda x: get_wordsegment(x))
df['content'] = df['content'].map(lambda x: test(x))
step += 1
step_name = '_wordsegment'
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
# -----------------------------


# ---------- STEP
# ---------- STEMMING --> generate the root form of inflected/desired words
# https://towardsdatascience.com/nlp-building-text-cleanup-and-preprocessing-pipeline-eba4095245a0


# function for stemming
def get_stem(text):
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]
    return text


# call function
df['content'] = df['content'].map(lambda x: get_stem(x))
step += 1
step_name = '_stemming'
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
# -----------------------------


# ---------- STEP
# ---------- LEMMATIZATION --> generate the root form of inflected/desired words; lemmatization is an advanced form of stemming
# https://towardsdatascience.com/nlp-building-text-cleanup-and-preprocessing-pipeline-eba4095245a0
# nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)
#
#
# # function to remove special characters
# def get_lem(text):
#     text = nlp(text)
#     text = [word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text]
#     return text
from nltk.stem.wordnet import WordNetLemmatizer
def get_lem(text):
    result = [WordNetLemmatizer().lemmatize(word) for word in text]
    return result

# call function
df['content'] = df['content'].map(lambda x: get_lem(x))
step += 1
step_name = '_lemmatization'
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
# -----------------------------


# ---------- STEP
# ---------- STOP WORDS
# https://towardsdatascience.com/basic-tweet-preprocessing-in-python-efd8360d529e
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# call function
df['content'] = df['content'].apply(lambda x: [item for item in x if item not in stop_words])
step += 1
step_name = '_stopwords'
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + step_name + '.csv', index=False, header=True)
# -----------------------------