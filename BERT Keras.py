import tensorflow as tf
from keras.layers import Layer, Embedding, Dropout, Dense, LayerNormalization, Input
from keras.models import Model
from keras.backend import softmax
import math
import numpy as np
from transformers import BertTokenizer



class PositionalEmbeddingTF(Layer):
    def __init__(self, d_model, max_len=128):
        super(PositionalEmbeddingTF, self).__init__()
        div_terms_sin = []
        div_terms_cos = []
        for pos in range(max_len):
            div_term_sin = tf.math.sin(pos / tf.pow(10000.0, tf.range(0, d_model, 2, dtype=tf.float32) / d_model * 2))
            div_term_cos = tf.math.cos(pos / tf.pow(10000.0, tf.range(1, d_model + 1, 2, dtype=tf.float32) / d_model * 2))
            div_terms_sin.append(div_term_sin)
            div_terms_cos.append(div_term_cos)
        pe = tf.reshape(tf.stack([div_terms_sin, div_terms_cos], axis=2), shape=(max_len, d_model))
        self.pe = tf.expand_dims(pe, axis=0)

    def call(self, x):
        return self.pe

class AddNormalizationTF(Layer):
    def __init__(self, **kwargs):
        super(AddNormalizationTF, self).__init__(**kwargs)
        self.layer_norm = LayerNormalization()  # Layer normalization layer
 
    def call(self, x, sublayer_x):
        # The sublayer input and output need to be of the same shape to be summed
        add = x + sublayer_x
 
        # Apply layer normalization to the sum
        return self.layer_norm(add)
    
class GELU(Layer):
    def call(self, inputs):
        return tf.keras.activations.gelu(inputs, approximate=True)

class FeedForwardTF(Layer):
    def __init__(self, d_ff, d_model, dropout=0.1):
        super(FeedForwardTF, self).__init__()

        self.fc1 = Dense(d_ff)
        self.fc2 = Dense(d_model)
        self.dropout = Dropout(dropout)
        self.activation = GELU()

    def call(self, inputs):
        out = self.activation(self.fc1(inputs))
        out = self.fc2(self.dropout(out))
        return out
    
class BERTEmbeddingTF(Layer):
    """
    :param vocab_size: total vocab size
    :param embed_size: embedding size of token embedding
    :param dropout: dropout rate
    """
    def __init__(self, vocab_size, embed_size, seq_len=64, dropout=0.1):
        super(BERTEmbeddingTF, self).__init__()
        self.embed_size = embed_size
        self.token = Embedding(vocab_size, embed_size, mask_zero=True)
        self.position = PositionalEmbeddingTF(d_model=embed_size, max_len=seq_len)
        self.dropout = Dropout(dropout)
        
    def call(self, sequence):
        token_embedding = self.token(sequence)
        positional_embedding = self.position(sequence)
        x = token_embedding + positional_embedding
        return self.dropout(x)

class DotProductAttentionTF(Layer):
    def __init__(self, **kwargs):
        super(DotProductAttentionTF, self).__init__(**kwargs)
 
    def call(self, queries, keys, values, d_k, mask=None):
        # Scoring the queries against the keys after transposing the latter, and scaling
        scores = tf.matmul(queries, keys, transpose_b=True) / math.sqrt(tf.cast(d_k, tf.float32))
 
        # Apply mask to the attention scores
        if mask is not None:
            scores += -1e9 * mask
 
        # Computing the weights by a softmax operation
        weights = softmax(scores)
 
        # Computing the attention by a weighted sum of the value vectors
        return tf.matmul(weights, values)
 
class MultiHeadAttentionTF(Layer):
    def __init__(self, h, d_k, d_v, d_model, **kwargs):
        super(MultiHeadAttentionTF, self).__init__(**kwargs)
        self.attention = DotProductAttentionTF()  
        self.heads = h 
        self.d_k = d_k 
        self.d_v = d_v  
        self.d_model = d_model  
        self.W_q = Dense(d_k)  
        self.W_k = Dense(d_k)  
        self.W_v = Dense(d_v)  
        self.W_o = Dense(d_model)  
 
    def reshape_tensor(self, x, heads, flag):
        if flag:
            # Tensor shape after reshaping and transposing: (batch_size, heads, seq_length, -1)
            x = tf.reshape(x, shape=(tf.shape(x)[0], tf.shape(x)[1], heads, -1))
            x = tf.transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping and transposing operations: (batch_size, seq_length, d_k)
            x = tf.transpose(x, perm=(0, 2, 1, 3))
            x = tf.reshape(x, shape=(tf.shape(x)[0], tf.shape(x)[1], self.d_k))
        return x
 
    def call(self, queries, keys, values, mask=None):
        q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
        k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
        v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)
        o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)
        
        output = self.reshape_tensor(o_reshaped, self.heads, False)
        
        return self.W_o(output)

class EncoderLayerTF(Layer):
    def __init__(self, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super(EncoderLayerTF, self).__init__(**kwargs)
        self.multihead_attention = MultiHeadAttentionTF(h, d_k, d_v, d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalizationTF()
        self.feed_forward = FeedForwardTF(d_ff, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalizationTF()
 
    def call(self, x, padding_mask, training):
        # Multi-head attention layer
        multihead_output = self.multihead_attention(x, x, x, padding_mask)
        multihead_output = self.dropout1(multihead_output, training=training)
        addnorm_output = self.add_norm1(x, multihead_output)
        feedforward_output = self.feed_forward(addnorm_output)
        feedforward_output = self.dropout2(feedforward_output, training=training)
 
        return self.add_norm2(addnorm_output, feedforward_output)

class BERTTF(Layer):
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate, **kwargs):
        super(BERTTF, self).__init__(**kwargs)
        self.input_embedding = BERTEmbeddingTF(vocab_size=vocab_size, embed_size=d_model, seq_len=sequence_length, dropout=rate)
        self.dropout = Dropout(rate)
        self.encoder_layer = [EncoderLayerTF(h, d_k, d_v, d_model, d_ff, rate) for _ in range(n)]
 
    def call(self, input_sentence, padding_mask, training):
        x = self.input_embedding(input_sentence)
        for i, layer in enumerate(self.encoder_layer):
            x = layer(x, padding_mask, training)
 
        return x
    
class BERTClassifierTF(Model):
    def __init__(self, bert: BERTTF, num_classes: int, dropout_rate: float = 0.1, **kwargs):
        super(BERTClassifierTF, self).__init__(**kwargs)
        self.bert = bert
        self.dropout = Dropout(dropout_rate)
        self.classifier = Dense(num_classes, activation='softmax')
 
    def call(self, input_sentence, padding_mask, training):
        x = self.bert(input_sentence, padding_mask, training)
        x = self.dropout(x, training=training)
        x = self.classifier(x[:, 0, :])
 
        return x
    
if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    sentences = ["I love to explore new technologies. I love AI", "This is an example sentence.", "This is a question?", "This is an answer."]

    input_id = []

    for sentence in sentences:
        # Tokenize the sentences and generate segment labels with a maximum length of 15
        encoded_inputs = tokenizer.encode_plus(
            sentence, add_special_tokens=True, max_length=400, truncation=True, padding='max_length'
        )
        input_id.append(encoded_inputs['input_ids'])
        
    # Convert input lists to NumPy arrays
    input_id = np.array(input_id)
     
    enc_vocab_size = 10000 # Vocabulary size for the encoder
    input_seq_length = 400  # Maximum length of the input sequence
    h = 4  # Number of self-attention heads
    d_k = 64  # Dimensionality of the linearly projected queries and keys
    d_v = 64  # Dimensionality of the linearly projected values
    d_ff = 2048  # Dimensionality of the inner fully connected layer
    d_model = 512  # Dimensionality of the model sub-layers' outputs
    n = 2  # Number of layers in the encoder stack
    dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers
    
    encoder = BERTTF(enc_vocab_size, input_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)