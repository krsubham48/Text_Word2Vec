import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from gensim.models import word2vec
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

data = pd.read_csv('/Users/kr_subham/Desktop/Word2Vec/labeledTrainData.tsv', delimiter='\t')
data.drop('id', axis=1, inplace=True)
print(data.head())

review_clean = []
stops = set(stopwords.words('english'))
def clean_review(raw_input):
    no_markups = BeautifulSoup(raw_input, 'html5lib')
    text = no_markups.get_text()
    only_text = re.sub('[^a-zA-Z]', ' ', text)
    lower_text = only_text.lower()
    tokens = lower_text.split()
    words = [w for w in tokens if not w in stops]
    return words

for i in data['review']:
    review_clean.append(clean_review(i))

print(review_clean[19][:5])

model = word2vec.Word2Vec(review_clean, min_count=1, size=400, workers=4)
print('Words similar to HAPPY:\n',  model.wv.most_similar('happy'))
print('Words similar to INTERESTING:\n',  model.wv.most_similar('interesting'))

def similar_words(model, word):
    
    arr = np.empty((0,400), dtype='f') #creates an empty array of size equalto the size of embeddings
    word_labels = [word] #initializes the word_labels list with the given word
    close_words = model.wv.similar_by_word(word) #a list of all the words similar to the given word
    
    arr = np.append(arr, np.array([model.wv[word]]), axis=0)
    for wrd_score in close_words: #for every word in close_words list
        wrd_vector = model.wv[wrd_score[0]] #this list stores the word embeddings of all the close_words
        word_labels.append(wrd_score[0]) #this list stores the labels of corresponding words stored
        arr = np.append(arr, np.array([wrd_vector]), axis=0) #each entry is appended into arr
        
    tsne = TSNE(n_components=2, random_state=0) #a 2 components t-SNE classifier is created
    Y = tsne.fit_transform(arr) #dimentionality is reduced for plotting
    x_coords = Y[:, 0] #x and y coordinates are separated
    y_coords = Y[:, 1] #in their respective lists
    plt.scatter(x_coords, y_coords) #a scatter plot is formed

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points') #labels are also displayed alongwith the plot
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()

similar_words(model, 'happy')
similar_words(model, 'interesting')