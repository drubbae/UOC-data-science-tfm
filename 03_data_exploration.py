import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


debug = True
test = True
step = 0
path_input = r'C:\Users\minions\Documents\data'
path_output = r'C:\Users\minions\Documents\data'


# ---------- STEP
# ---------- READ DATA
if test:
    df = pd.read_csv(path_input + r'\02_data_test.csv')
    if debug:
        df = df.filter(['content'])
else:
    df = pd.read_csv(path_input + r'\02_data.csv')
    if debug:
        df = df.filter(['content'])
# -----------------------------


# ---------- STEP
# ---------- WORDCLOUD
# create a wordcloud object
wordcloud = WordCloud(background_color="white", max_words=2500, contour_width=3, contour_color='steelblue')
# generate a wordcloud
wordcloud.generate(df['content'])
# visualize the wordcloud
wordcloud.to_image()
# make figure to plot
plt.figure()
# plot words
plt.imshow(wordcloud, interpolation="bilinear")
# remove axes
plt.axis("off")
# show the result
plt.show()
# -----------------------------