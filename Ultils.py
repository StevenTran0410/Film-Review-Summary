import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Input, PReLU
from keras.initializers import RandomUniform

summary = ["Negative","Positive"]

def model_building(name):
    if name == "Dense":
        model = Sequential()

        model.add(Input(100))

        model.add(Dense(256))
        model.add(PReLU(alpha_initializer=RandomUniform(minval=0.1, maxval=0.5)))  # Random uniform initialization
        model.add(Dense(128))
        model.add(PReLU(alpha_initializer=RandomUniform(minval=0.1, maxval=0.5)))  
        model.add(Dense(64))
        model.add(PReLU(alpha_initializer=RandomUniform(minval=0.1, maxval=0.5)))  
        model.add(Dense(32))
        model.add(PReLU(alpha_initializer=RandomUniform(minval=0.1, maxval=0.5)))  

        model.add(Dropout(0.2))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        model.load_weights("./Weights/Sentiment Analysis Dense.h5")
            
    return model

def model_word2vec():
    return Word2Vec.load("./Weights/word2vec_model.bin")

def word2vec(review, word2vec_model):
    # Tokenize the text data
    tokenized_reviews = review.split()  # Split the string into tokens

    # Convert text to word embeddings
    embeddings = [word2vec_model.wv[word] for word in tokenized_reviews if word in word2vec_model.wv]

    # Obtain fixed-size representation by averaging
    averaged_reviews = np.mean(embeddings, axis=0)

    # Convert to numpy array
    averaged_reviews = np.array([averaged_reviews])
    
    return averaged_reviews

def clean_text(text):
    # Remove <br> tags
    text = re.sub(r'<br\s*/?>', ' ', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    stop_words.update(["mr", "ms", "mrs", "dr", "film", "movie", "really"])  # Add more stop words as needed                          film and movie was added due to the frequent they appear in both negative and positive
    text = ' '.join(word for word in text.split() if word not in stop_words)

    # Remove 1-2 length words
    text = ' '.join(word for word in text.split() if len(word) > 2)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())

    return text

def detect(input, model, word2vec_model):
    text = clean_text(input)
    text = word2vec(text, word2vec_model)
    result = np.argmax(model.predict(text))
    
    return summary[result]
    
    
