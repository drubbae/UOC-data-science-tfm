# https://towardsdatascience.com/nlp-building-text-cleanup-and-preprocessing-pipeline-eba4095245a0
from langdetect import detect
import pandas as pd
from wordsegment import load, segment
import nltk
nltk.download('wordnet')
import re
import emoji
import string
import unicodedata
import time
print('package --> pandas --> version --> ' + pd.__version__)
print('package --> re     --> version --> ' + re.__version__)
print('package --> ntlk   --> version --> ' + nltk.__version__)


debug = True
test = True
step = 0
path_input = r'C:\Users\minions\Documents\data'
path_output = r'C:\Users\minions\Documents\data'


# ----------- STEP
# ----------- READ/LOAD PARSED DATA
time_start = time.time()
if test:
    df = pd.read_csv(path_input + r'\01_data_test.csv')
else:
    df = pd.read_csv(path_input + r'\01_data.csv')
step_name = 'read'
if debug:
    df = df.filter(['content'])
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
time_end = time.time()
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling-' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))


# ---------- STEP
# ---------- LOWERCASE
time_start = time.time()
df['content'] = df['content'].map(lambda x: str(x).lower())
step += 1
step_name = 'lowercase'
if debug:
    df = df.filter(['content'])
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
time_end = time.time()
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling-' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))


# ---------- STEP
# ---------- REPLACE WEBSITES BY WHITESPACES
# (https://emailregex.com/)
time_start = time.time()
# extract websites, and save them into a new field named 'website'
re_pattern = r'((http[s]?://)|(www))(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
df['website'] = df['content'].apply(lambda x: re.findall(re_pattern, str(x)))
# remove websites
df['content'] = df['content'].apply(lambda x: re.sub(re_pattern, ' ', str(x)))
step += 1
step_name = 'remove_websites'
if debug:
    df = df.filter(['content'])
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
time_end = time.time()
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling-' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))


# ---------- STEP
# ---------- REPLACE EMAILS BY WHITESPACES
time_start = time.time()
# extract emails, and save them into a new field named 'email'
re_pattern = r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'
df['email'] = df['content'].apply(lambda x: re.findall(re_pattern, x))
# remove emails
df['content'] = df['content'].apply(lambda x: re.sub(re_pattern, ' ', x))
step += 1
step_name = 'remove_emails'
if debug:
    df = df.filter(['content'])
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
time_end = time.time()
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling-' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))


# ---------- STEP
# ---------- TREAT HASHTAGS
# (https://towardsdatascience.com/basic-tweet-preprocessing-in-python-efd8360d529e)
time_start = time.time()
# extract hashtags, and save them into a new field named 'hashtag'
re_pattern = r'#(\w+)'
df['hashtag'] = df['content'].apply(lambda x: re.findall(re_pattern, x))
# remove the hashtag identifier symbol
re_pattern = r'(#)'
df['content'] = df['content'].apply(lambda x: re.sub(re_pattern, ' ', x))
step += 1
step_name = 'remove_hashtags'
if debug:
    df = df.filter(['content'])
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
time_end = time.time()
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling-' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))


# ---------- STEP
# ---------- REPLACE MENTIONS BY WHITESPACES
time_start = time.time()
# extract mentions, and save them into a new field named 'mention'
re_pattern = r'@(\w+)'
df['mention'] = df['content'].apply(lambda x: re.findall(re_pattern, x))
# remove the mention identifier symbol
re_pattern = r'(@\w+)'
df['content'] = df['content'].apply(lambda x: re.sub(re_pattern, ' ', x))
step += 1
step_name = 'remove_mentions'
if debug:
    df = df.filter(['content'])
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
time_end = time.time()
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling-' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))


# ---------- STEP
# ---------- REPLACE COLON BY WHITESPACES
time_start = time.time()
# remove punctuation
re_pattern = r'(:)'
df['content'] = df['content'].apply(lambda x: re.sub(re_pattern, ' ', str(x)))
step += 1
step_name = 'remove_colons'
if debug:
    df = df.filter(['content'])
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
time_end = time.time()
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling-' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))


# ---------- STEP
# ---------- REPLACE EMOJIS BY WHITESPACES
time_start = time.time()
# replace emoji by its short name between ':' --> e.g. --> :face_blowing_a_kiss:
df['content'] = df['content'].map(lambda x: emoji.demojize(x))
step += 1
step_name = 'replace_emojis'
if debug:
    df = df.filter(['content'])
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
time_end = time.time()
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling-' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))

time_start = time.time()
# remove emoji's short name between ':'
re_pattern = r'\:(.*?)\:'
df['content'] = df['content'].apply(lambda x: re.sub(re_pattern, ' ', x))
step += 1
step_name = 'remove_emojis'
if debug:
    df = df.filter(['content'])
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
time_end = time.time()
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling-' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))


# ---------- STEP
# ---------- TREAT CONTRACTIONS
import contractions

time_start = time.time()
df['content'] = df['content'].apply(lambda x: contractions.fix(x))
step += 1
step_name = 'expand_contractions'
if debug:
    df = df.filter(['content'])
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
time_end = time.time()
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling-' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))


# ---------- STEP
# ---------- ACCENTED
# (https://towardsdatascience.com/nlp-building-text-cleanup-and-preprocessing-pipeline-eba4095245a0)
# remove accented characters


# function to remove accented characters
def remove_accented_chars(text):
   new_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
   return new_text


time_start = time.time()
# call function to remove accented characters (remove_accented_chars)
df['content'] = df['content'].map(lambda x: remove_accented_chars(x))
step += 1
step_name = 'remove_accents'
if debug:
    df = df.filter(['content'])
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
time_end = time.time()
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling-' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))


# ---------- STEP
# ---------- REPLACE NUMBERS BY WHITESPACES
time_start = time.time()
re_pattern = r'[^a-zA-z.,!?/:;\"\'\s]'
df['content'] = df['content'].apply(lambda x: re.sub(re_pattern, ' ', x))
step += 1
step_name = 'remove_numbers'
if debug:
    df = df.filter(['content'])
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
time_end = time.time()
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling-' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))


# ---------- STEP
# ---------- REMOVE SPECIAL CHARACTERS
time_start = time.time()
# pattern to keep
re_pattern = r'[^a-zA-z0-9.,!?/:;\"\'\s]'
df['content'] = df['content'].apply(lambda x: re.sub(re_pattern, ' ', x))
step += 1
step_name = 'remove_special_characters'
if debug:
    df = df.filter(['content'])
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
time_end = time.time()
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling-' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))


# ---------- STEP
# ---------- REMOVE PUNCTUATION
time_start = time.time()
df['content'] = df['content'].apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))
step += 1
step_name = 'remove_punctuation'
if debug:
    df = df.filter(['content'])
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
time_end = time.time()
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling-' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))


# ---------- STEP
# ---------- REMOVE EXTRA WHITESPACES
time_start = time.time()
re_pattern = r' +'
df['content'] = df['content'].apply(lambda x: re.sub(re_pattern, ' ', x))
step += 1
step_name = 'remove_extra_whitespaces'
if debug:
    df = df.filter(['content'])
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
time_end = time.time()
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling-' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))


# ---------- STEP
# ---------- LANGUAGE
# (https://medium.com/1001-nights-in-data-science/three-methods-of-detecting-language-of-textual-data-2c7cc41033b1)
# function to detect language
def detect_language(text):
    try:
        language_detected = detect(text)
    except:
        language_detected = ''
    return language_detected


time_start = time.time()
# call function to detect language (detect_language)
df['language'] = df['content'].map(lambda x: detect_language(x))
step += 1
step_name = 'detect_language'
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
time_end = time.time()
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling-' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))


# ---------- STEP
# ---------- SELECT ONLY LANGUAGE (ENGLISH)
time_start = time.time()
df = df[df['language'] == 'en']
step += 1
step_name = 'select_english'
if debug:
    df = df.filter(['content'])
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
time_end = time.time()
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling-' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))


# ---------- STEP
# ---------- TOKENIZATION
from nltk.tokenize import word_tokenize
nltk.download('punkt')

time_start = time.time()
df['content'] = df['content'].map(lambda x: word_tokenize(x))
step += 1
step_name = 'tokenization'
if debug:
    df = df.filter(['content'])
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
time_end = time.time()
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling-' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))


# ---------- STEP
# ---------- TREAT SPELLING ERRORS
# (https://medium.com/analytics-vidhya/working-with-twitter-data-b0aa5419532) (https://github.com/fsondej/autocorrect)
from autocorrect import Speller
spell = Speller()


def treat_spelling_errors(text):
    text = [spell(w) for w in text]
    return text


time_start = time.time()
# call function
df['content'] = df['content'].map(lambda x: treat_spelling_errors(x))
step += 1
step_name = 'spelling_errors'
if debug:
    df = df.filter(['content'])
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
time_end = time.time()
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling-' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))


# ---------- STEP
# ---------- WORDSEGMENT
# (https://medium.com/analytics-vidhya/working-with-twitter-data-b0aa5419532)
# function to perform word segmentation
load()


def get_word_segment(text):
    text = [segment(word) for word in text]
    return text


# function to adapt the results obtained after executing word segmentation
# word segmentation returns a list of lists
def adapt_list(list_of_lists):
    result = [item for elem in list_of_lists for item in elem]
    return result


time_start = time.time()
# call function
df['content'] = df['content'].map(lambda x: get_word_segment(x))
df['content'] = df['content'].map(lambda x: adapt_list(x))
step += 1
step_name = 'wordsegment'
if debug:
    df = df.filter(['content'])
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
time_end = time.time()
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling-' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))


# ---------- STEP
# ---------- LEMMATIZATION --> generate the root form of inflected/desired words; lemmatization is an advanced form of stemming
# (https://towardsdatascience.com/nlp-building-text-cleanup-and-preprocessing-pipeline-eba4095245a0)
from nltk.stem.wordnet import WordNetLemmatizer


# function to perform lemmatization
def get_lem(text):
    result = [WordNetLemmatizer().lemmatize(word) for word in text]
    return result


time_start = time.time()
# call function
df['content'] = df['content'].map(lambda x: get_lem(x))
step += 1
step_name = 'lemmatization'
if debug:
    df = df.filter(['content'])
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
time_end = time.time()
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling-' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))


# ---------- STEP
# ---------- REMOVE STOP WORDS
# (https://towardsdatascience.com/basic-tweet-preprocessing-in-python-efd8360d529e)
import csv
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')
with open(r'C:\Users\minions\Documents\data\stopwords_extend.csv', newline='') as f:
    reader = csv.reader(f)
    stopwords_extend = list(reader)
stop_words.extend(stopwords_extend)

time_start = time.time()
df['content'] = df['content'].apply(lambda x: [item for item in x if item not in stop_words])
step += 1
step_name = 'stopwords'
if debug:
    df = df.filter(['content'])
if test:
    df.to_csv(path_output + r'\02_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
time_end = time.time()
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling-' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))


# ---------- LAST STEP
# ---------- SAVE PRE-PROCESSED DATA
if test:
    df.to_csv(path_output + r'\02_data_test_.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\02_data_.csv', index=False, header=True)