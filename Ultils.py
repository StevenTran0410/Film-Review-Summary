import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, PReLU, LSTM, Masking
from keras.initializers import RandomUniform
from keras_preprocessing.sequence import pad_sequences

summary = ["Negative","Positive"]

def model_building(name):
    if name == "Dense":
        model = Sequential()

        model.add(Input(200))
        model.add(Dense(256))
        model.add(PReLU(alpha_initializer=RandomUniform(minval=0.1, maxval=0.5)))  # Random uniform initialization
        model.add(Dense(512))
        model.add(PReLU(alpha_initializer=RandomUniform(minval=0.1, maxval=0.5)))
        model.add(Dense(256))
        model.add(PReLU(alpha_initializer=RandomUniform(minval=0.1, maxval=0.5)))
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
        
        word2vec_model = Word2Vec.load("./Weights/word2vec_model.bin")
    elif name == "LSTM":
        model = Sequential()
        
        model.add(Input(shape=(250, 75)))  # Specify the input shape as a tuple
        model.add(Masking(mask_value=0))
        model.add(LSTM(64, activation='tanh', dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(64))
        model.add(PReLU(alpha_initializer=RandomUniform(minval=0.1, maxval=0.5)))  
        model.add(Dense(32))
        model.add(PReLU(alpha_initializer=RandomUniform(minval=0.1, maxval=0.5)))  
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.load_weights("./Weights/Sentiment Analysis RNN LSTM.h5")
        
        word2vec_model = Word2Vec.load("./Weights/word2vec_model_RNN.bin")

            
    return model, word2vec_model


def word2vec(review, word2vec_model, selected_model):
    if selected_model == 1:
        # Tokenize the text data
        tokenized_reviews = review.split()  # Split the string into tokens

        # Convert text to word embeddings
        embeddings = [word2vec_model.wv[word] for word in tokenized_reviews if word in word2vec_model.wv]

        # Obtain fixed-size representation by averaging
        averaged_reviews = np.array(np.mean(embeddings, axis=0))
        
        return averaged_reviews.reshape(1, 200)
    elif selected_model == 2:
        # Tokenize the text data
        tokenized_review = review.split()

        # Convert text to word embeddings
        embeddings = np.array([word2vec_model.wv[word] for word in tokenized_review if word in word2vec_model.wv])
        embeddings = embeddings.reshape((1,) + embeddings.shape)
        sequence = pad_sequences(embeddings, maxlen=250, padding='post', truncating='post', dtype=np.float32)
        
        return sequence

def clean_text(text, selected_model):
    if selected_model == 1:
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
    elif selected_model == 2:
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
        stop_words.update(["mr", "ms", "mrs", "dr", "film", "movie", "really", "one"])  # Add more stop words as needed                          film and movie was added due to the frequent they appear in both negative and positive
        text = ' '.join(word for word in text.split() if word.lower() not in stop_words or word.lower() == "not")

        # Remove 1-2 length words
        text = ' '.join(word for word in text.split() if len(word) > 2)

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())

        return text

def detect(input, model, word2vec_model, selected_model):
    text = clean_text(input, selected_model)
    text = word2vec(text, word2vec_model, selected_model)
    result = np.argmax(model.predict(text))
    
    return summary[result]
    
    
