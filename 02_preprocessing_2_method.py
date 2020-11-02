# https://towardsdatascience.com/nlp-building-text-cleanup-and-preprocessing-pipeline-eba4095245a0
from langdetect import detect
import pandas as pd
import re
import emoji
import string
import unicodedata
print('package --> pandas --> version --> ' + pd.__version__)
print('package --> re     --> version --> ' + re.__version__)

debug = False

# ---------- STEP 01 ----------
# load parsed data1
df = pd.read_csv(r'D:\master\data science\semestre 4\M2.979 - tfm\data\02_raw_influencers_2_method.csv')
# -----------------------------


# STEP DATA EXPLORATION
# null data
for col in df.columns:
    print(col, df[col].isnull().sum())


# ---------- STEP 02 ----------
# replace emoji by its short name between ':' --> e.g. --> :face_blowing_a_kiss:
df['content'] = df['content'].map(lambda x: emoji.demojize(str(x)))
#df.to_csv(r'D:\master\data science\semestre 4\M2.979 - tfm\data\02_raw_influencers_2_method_test_01.csv', index=False, header=True)
# remove emoji's short name between ':'
re_pattern = r'\:(.*?)\:'
df['content'] = df['content'].apply(lambda x: re.sub(re_pattern, ' ', x))
#df.to_csv(r'D:\master\data science\semestre 4\M2.979 - tfm\data\02_raw_influencers_2_method_test_02.csv', index=False, header=True)
# -----------------------------


# ---------- STEP 03 ----------
# https://towardsdatascience.com/basic-tweet-preprocessing-in-python-efd8360d529e
# extract hashtags, and save them into a new field named 'hashtag'
re_pattern = r'#(\w+)'
df['hashtag'] = df['content'].apply(lambda x: re.findall(re_pattern, x))
# remove the hashtag identifier symbol
re_pattern = r'(#)'
df['content'] = df['content'].apply(lambda x: re.sub(re_pattern, ' ', x))
#df.to_csv(r'D:\master\data science\semestre 4\M2.979 - tfm\data\02_raw_influencers_2_method_test_03.csv', index=False, header=True)
# -----------------------------


# ---------- STEP 04 ----------
# extract emails, and save them into a new field named 'email'
re_pattern = r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'
df['email'] = df['content'].apply(lambda x: re.findall(re_pattern, x))
# remove emails
df['content'] = df['content'].apply(lambda x: re.sub(re_pattern, ' ', x))
#df.to_csv(r'D:\master\data science\semestre 4\M2.979 - tfm\data\02_raw_influencers_2_method_test_04.csv', index=False, header=True)
# -----------------------------


# ---------- STEP 05 ----------
# extract websites, and save them into a new field named 'website'
re_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
df['website'] = df['content'].apply(lambda x: re.findall(re_pattern, x))
# remove websites
df['content'] = df['content'].apply(lambda x: re.sub(re_pattern, ' ', x))
#df.to_csv(r'D:\master\data science\semestre 4\M2.979 - tfm\data\02_raw_influencers_2_method_test_05.csv', index=False, header=True)
# -----------------------------


# ---------- STEP 06 ----------
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
#df.to_csv(r'D:\master\data science\semestre 4\M2.979 - tfm\data\02_raw_influencers_2_method_test_06.csv', index=False, header=True)
# -----------------------------


# ----------STEP 07 ----------
# lowercase the text
df['content'] = df['content'].map(lambda x: x.lower())
#df.to_csv(r'D:\master\data science\semestre 4\M2.979 - tfm\data\02_raw_influencers_2_method_test_07.csv', index=False, header=True)
# -----------------------------


# ---------- STEP 08 ----------
# https://towardsdatascience.com/nlp-building-text-cleanup-and-preprocessing-pipeline-eba4095245a0
# remove accented characters


# function to remove accented characters
def remove_accented_chars(text):
    new_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return new_text


# call function to remove accented characters (remove_accented_chars)
df['content'] = df['content'].map(lambda x: remove_accented_chars(x))
#df.to_csv(r'D:\master\data science\semestre 4\M2.979 - tfm\data\02_raw_influencers_2_method_test_08.csv', index=False, header=True)
# -----------------------------


# ---------- STEP 09 ----------
# remove numbers
re_pattern = r'[^a-zA-z.,!?/:;\"\'\s]'
df['content'] = df['content'].apply(lambda x: re.sub(re_pattern, ' ', x))
#df.to_csv(r'D:\master\data science\semestre 4\M2.979 - tfm\data\02_raw_influencers_2_method_test_09.csv', index=False, header=True)
# -----------------------------


# ---------- STEP 10 ----------
# remove punctuation
df['content'] = df['content'].apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))
#df.to_csv(r'D:\master\data science\semestre 4\M2.979 - tfm\data\02_raw_influencers_2_method_test_10.csv', index=False, header=True)
# -----------------------------


# ---------- STEP 11 ----------
# remove extra spaces
re_pattern = r' +'
df['content'] = df['content'].apply(lambda x: re.sub(re_pattern, ' ', x))
df.to_csv(r'D:\master\data science\semestre 4\M2.979 - tfm\data\02_raw_influencers_2_method_11.csv', index=False, header=True)
# -----------------------------