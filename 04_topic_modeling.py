# https://medium.com/analytics-vidhya/topic-modeling-using-gensim-lda-in-python-48eaa2344920
import csv
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import spacy
import time
from pprint import pprint

with open(r'C:\Users\minions\Documents\data\stopwords_extend.csv', newline='') as f:
    reader = csv.reader(f)
    stopwords_extend = list(reader)

stop_words = stopwords.words('english')
stop_words.extend(stopwords_extend)


debug = True
test = True
step = 0
path_input = r'C:\Users\minions\Documents\data'
path_output = r'C:\Users\minions\Documents\data'


# ----------- STEP
# ----------- READ/LOAD DATA
time_start = time.time()
if test:
    df = pd.read_csv(path_input + r'\02_data_test.csv')
else:
    df = pd.read_csv(path_input + r'\02_data.csv')
step_name = 'read'
if debug:
    df = df.filter(['content'])
if test:
    df.to_csv(path_output + r'\03_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
else:
    df.to_csv(path_output + r'\03_data_' + str(step).zfill(2) + '_' + step_name + '.csv', index=False, header=True)
time_end = time.time()
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling --> ' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))
# -----------------------------


# ----------- STEP
# ----------- SPLIT SENTENCES TO WORDS
def sent_to_words(sentences):
    for sentence in sentences:
        yield(simple_preprocess(str(sentence), deacc=True))


time_start = time.time()
data_words = list(sent_to_words(df['content']))
step += 1
step_name = 'split_sentences_to_words'
if test:
    np.savetxt(path_output + r'\03_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv',
               data_words,
               delimiter=',',
               fmt='% s')
else:
    np.savetxt(path_output + r'\03_data_' + str(step).zfill(2) + '_' + step_name + '.csv',
               data_words,
               delimiter=',',
               fmt='% s')
time_end = time.time()
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling --> ' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))
# -----------------------------


# ----------- STEP
# ----------- BUILD N-GRAMS
time_start = time.time()
# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
# Faster way to get a sentence formatted as a bigram or trigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
time_end = time.time()
step += 1
step_name = 'build_ngrams'
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling --> ' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))
# -----------------------------


# # Define function for trigrams
# def make_trigrams(texts):
#     return [trigram_mod[bigram_mod[doc]] for doc in texts]


# ----------- STEP
# ----------- REMOVE STOPWORDS
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


time_start = time.time()
data_words_nostops = remove_stopwords(data_words)
step += 1
step_name = 'stopwords'
if test:
    np.savetxt(path_output + r'\03_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv',
               data_words,
               delimiter=',',
               fmt='% s')
else:
    np.savetxt(path_output + r'\03_data_' + str(step).zfill(2) + '_' + step_name + '.csv',
               data_words,
               delimiter=',',
               fmt='% s')
time_end = time.time()
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling --> ' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))
# -----------------------------


# ----------- STEP
# ----------- FORM BIGRAMS
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


time_start = time.time()
data_words_bigrams = make_bigrams(data_words_nostops)
step += 1
step_name = 'bigrams'
if test:
    np.savetxt(path_output + r'\03_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv',
               data_words,
               delimiter=',',
               fmt='% s')
else:
    np.savetxt(path_output + r'\03_data_' + str(step).zfill(2) + '_' + step_name + '.csv',
               data_words,
               delimiter=',',
               fmt='% s')
time_end = time.time()
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling --> ' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))
# -----------------------------


# ----------- STEP
# ----------- LEMMATIZATION
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(' '.join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


time_start = time.time()
# initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
# lemmatize keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
step += 1
step_name = 'lemmatization'
if test:
    np.savetxt(path_output + r'\03_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv',
               data_words,
               delimiter=',',
               fmt='% s')
else:
    np.savetxt(path_output + r'\03_data_' + str(step).zfill(2) + '_' + step_name + '.csv',
               data_words,
               delimiter=',',
               fmt='% s')
time_end = time.time()
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling --> ' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))
# -----------------------------


# ----------- STEP
# ----------- BUILD CORPORA
time_start = time.time()
step += 1

# create dictionary
id2word = corpora.Dictionary(data_lemmatized)
step_name = 'id2word'
if test:
    np.savetxt(path_output + r'\03_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv',
               data_words,
               delimiter=',',
               fmt='% s')
else:
    np.savetxt(path_output + r'\03_data_' + str(step).zfill(2) + '_' + step_name + '.csv',
               data_words,
               delimiter=',',
               fmt='% s')

# create corpus
texts = data_lemmatized
step_name = 'texts'
if test:
    np.savetxt(path_output + r'\03_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv',
               data_words,
               delimiter=',',
               fmt='% s')
else:
    np.savetxt(path_output + r'\03_data_' + str(step).zfill(2) + '_' + step_name + '.csv',
               data_words,
               delimiter=',',
               fmt='% s')

# term document frequency (TDF)
corpus = [id2word.doc2bow(text) for text in texts]
step_name = 'corpus'
if test:
    np.savetxt(path_output + r'\03_data_test_' + str(step).zfill(2) + '_' + step_name + '.csv',
               data_words,
               delimiter=',',
               fmt='% s')
else:
    np.savetxt(path_output + r'\03_data_' + str(step).zfill(2) + '_' + step_name + '.csv',
               data_words,
               delimiter=',',
               fmt='% s')

time_end = time.time()
if debug:
    hour, rem = divmod(time_end - time_start, 3600)
    minute, second = divmod(rem, 60)
    print('prepare-data-for-topic-modeling --> ' + step_name)
    print('time elapsed {:0>2}:{:0>2}:{:05.2f}'.format(int(hour), int(minute), second))
    print('time start ' + str(time.ctime(int(time_start))))
    print('time end ' + str(time.ctime(int(time_end))))
# -----------------------------


# ----------- STEP
# ----------- BUILD TOPIC MODEL
# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=10,
                                            random_state=111, #for reproducibility
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha=10,
                                            eta='auto',
                                            iterations=400,
                                            per_word_topics=True)

# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

# Interpretability and goodness of the model
# For this purpose we will now compute some statistical measures to estimate goodness of the models
# First up is perplexity. it is a statistical measure of how well a probability model is capable of predicting a sample.
# When compared across multiple models with different number of topics, the model with lowest perplexity score is chosen .
# But a lot of times the model with best possible perplexity does not result in human interpretable models
# So another set of measures named as a coherent scores are introduced.
# Broadly, this measure assigns a ascore to a topic by measuring the degree of semantic similarity between most frequent words in a topic
# It offers better interpretability of the model

# compute perplexity
print('Perplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. Lower value is preferred.

# # compute coherence score
# coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
# coherence_lda = coherence_model_lda.get_coherence()
# print('Coherence Score: ', coherence_lda)
#
# # compute coherence score using UMass
# coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='u_mass')
# coherence_lda = coherence_model_lda.get_coherence()
# print('Coherence Score u_mass: ', coherence_lda)
# -----------------------------


import pyLDAvis
import pyLDAvis.gensim
#pyLDAvis.enable_notebook()
lda_display = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, sort_topics=False)
pyLDAvis.save_html(lda_display, r'C:\Users\minions\Documents\data\LDAVisualization.html')