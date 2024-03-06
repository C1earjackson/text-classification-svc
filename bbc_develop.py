import os
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from gensim.models import Word2Vec
from sklearn.svm import SVC
from xgboost import XGBClassifier

# load_data function
def load_data(data_dir):
    categories = ['business', 'entertainment', 'politics', 'sport', 'tech']
    data = []
    for cat in categories:
        path = os.path.join(data_dir, cat)  # combine data_dir and category
        files = os.listdir(path)  # extract txt files to a list
        for file in files:
            with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
                text = f.read()
                data.append((text, cat))
    return pd.DataFrame(data, columns=['text', 'category'])

# Load data
df = load_data('bbc')

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    lemmatized = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    return ' '.join(lemmatized)

# Preprocess text data
df['processed_text'] = df['text'].apply(preprocess)

# TF-IDF feature extraction
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
tfidf_features = tfidf_vectorizer.fit_transform(df['processed_text'])

# Word2Vec feature extraction
sentences = [row.split() for row in df['processed_text']]
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
word2vec_model.train(sentences, total_examples=len(sentences), epochs=10)

# Creating a function to average Word2Vec vectors for a document
def document_vector(model, doc):
    doc = [word for word in doc if word in model.wv.key_to_index]
    return np.mean(model.wv[doc], axis=0) if doc else np.zeros(model.vector_size)
word2vec_features = np.array([document_vector(word2vec_model, doc) for doc in sentences])

# Word Frequency feature extraction using CountVectorizer
count_vectorizer = CountVectorizer()
count_features = count_vectorizer.fit_transform(df['processed_text']).toarray()

# Combining TF-IDF, Word2Vec, and Word Frequency features
X = np.hstack((tfidf_features.toarray(), word2vec_features, count_features))

# Dimensionality Reduction with PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X)

# Labels
y = df['category'].factorize()[0]

# Splitting the dataset into training, development, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_pca, y, test_size=0.4, random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Model Training on the Training Set
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Evaluate model on the Development Set
y_dev_pred01 = model.predict(X_dev)
print(" XGB Development Set Evaluation:")
print(classification_report(y_dev, y_dev_pred01))

y_dev_pred02 = classifier.predict(X_dev)
print(" SVC Development Set Evaluation:")
print(classification_report(y_dev, y_dev_pred02))

# Final Evaluation on Test Set (should only be done once model is fully tuned)
y_test_pred = classifier.predict(X_test)
print("Test Set Evaluation:")
print(classification_report(y_test, y_test_pred))
