import tensorflow as tf
from keras.layers import Layer, Embedding, Add, Dropout, Dense, LayerNormalization, Input, MultiHeadAttention, Flatten, ReLU
from keras.models import Model, Sequential
import numpy as np
from transformers import BertTokenizer
from keras import backend as K

def star_mae(y_true, y_pred):
    true_star = K.sum(y_true * K.arange(0, 5, dtype="float32"), axis=-1)
    pred_star = K.sum(y_pred * K.arange(0, 5, dtype="float32"), axis=-1)
    mae = K.mean(K.abs(true_star - pred_star))
    return mae

def PositionalEmbeddingTF(d_model, max_len=128):
    div_terms_sin = []
    div_terms_cos = []
    for pos in range(max_len):
        div_term_sin = tf.math.sin(pos / tf.pow(10000.0, tf.range(0, d_model, 2, dtype=tf.float32) / d_model * 2))
        div_term_cos = tf.math.cos(pos / tf.pow(10000.0, tf.range(1, d_model + 1, 2, dtype=tf.float32) / d_model * 2))
        div_terms_sin.append(div_term_sin)
        div_terms_cos.append(div_term_cos)
    pe = tf.reshape(tf.stack([div_terms_sin, div_terms_cos], axis=2), shape=(max_len, d_model))
    return tf.expand_dims(pe, axis=0)
    
# class PositionalEmbeddingTF(Layer):
#     def __init__(self, d_model, max_len=128):
#         super(PositionalEmbeddingTF, self).__init__()
#         div_terms_sin = []
#         div_terms_cos = []
#         for pos in range(max_len):
#             div_term_sin = tf.math.sin(pos / tf.pow(10000.0, tf.range(0, d_model, 2, dtype=tf.float32) / d_model * 2))
#             div_term_cos = tf.math.cos(pos / tf.pow(10000.0, tf.range(1, d_model + 1, 2, dtype=tf.float32) / d_model * 2))
#             div_terms_sin.append(div_term_sin)
#             div_terms_cos.append(div_term_cos)
#         pe = tf.reshape(tf.stack([div_terms_sin, div_terms_cos], axis=2), shape=(max_len, d_model))
#         self.pe = tf.expand_dims(pe, axis=0)

#     def call(self, x):
#         return self.pe
    
#     def get_config(self):
#         return super().get_config().update({"d_model": self.d_model, "max_len": self.max_len})
    
class BERTEmbeddingTF(Layer):
    """
    :param vocab_size: total vocab size
    :param embed_size: embedding size of token embedding
    :param dropout: dropout rate
    """
    def __init__(self, seq_len, vocab_size, embed_size, dropout=0.1):
        super(BERTEmbeddingTF, self).__init__()
        self.embed_size = embed_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.token = Embedding(vocab_size, embed_size, mask_zero=True)
        self.position = PositionalEmbeddingTF(d_model=embed_size, max_len=seq_len)
        self.dropout = Dropout(dropout)
        
    def call(self, sequence):
        token_embedding = self.token(sequence)
        positional_embedding = self.position
        x = token_embedding + positional_embedding
        return self.dropout(x)
    
    def compute_mask(self, *args, **kwargs):
        return self.token.compute_mask(*args, **kwargs)
    
    def get_config(self):
        return super().get_config().update({"vocab_size": self.vocab_size, "embed_size": self.embed_size, "seq_len": self.seq_len, "position": self.position})

def FeedForwardTF(input_shape, d_model, d_ff, dropout=0.1, prefix='ff'):
    inputs = Input(shape=input_shape, dtype=tf.float32, name=f'{prefix}_input')
    x = Dense(d_ff, name = f'{prefix}_ff1')(inputs)
    x = Dense(d_model, name = f'{prefix}_ff2')(x)
    x = ReLU(name = f'{prefix}_relu')(x)
    x = Dropout(dropout, name = f'{prefix}_dropout')(x)
    x = Add(name = f'{prefix}_add')([inputs, x]) 
    x = LayerNormalization(name = f'{prefix}_ln')(x)
    
    model = Model(inputs, x, name = f'{prefix}_model')
    return model

def MultiHeadAttentionTF(input_shape, prefix = "att", **kwargs):
    inputs = Input(input_shape, dtype=tf.float32, name=f'{prefix}_input')
    attention = MultiHeadAttention(name=f"{prefix}_attn1", **kwargs)
    x = attention(query=inputs, value=inputs, key=inputs)
    x = Add(name = f'{prefix}_add')([inputs, x])
    x = LayerNormalization(name = f'{prefix}_ln')(x)
    
    model = Model(inputs, x, name = f'{prefix}_model')
    return model

def EncoderLayerTF(input_shape, d_k, d_ff, dropout=0.1, prefix="enc", **kwargs):
    model = Sequential()
    
    model.add(Input(input_shape, dtype=tf.float32, name=f'{prefix}_input'))
    model.add(MultiHeadAttentionTF(input_shape, prefix=f"{prefix}_attn1", key_dim = d_k, **kwargs))
    model.add(Dropout(dropout, name = f'{prefix}_dropout1'))
    model.add(FeedForwardTF(input_shape, d_k, d_ff, prefix = f'{prefix}_ff'))
    
    return model

def BERTTF(num_layers, num_heads, seq_len, d_k, d_ff, vocab_size, num_class, dropout = 0.1, name = "BERTTF"):
    embed_shape = (seq_len, d_k)    
    inputs = Input((seq_len,), dtype=tf.int32, name = f'{name}_input')
    emebedding = BERTEmbeddingTF(seq_len, vocab_size, d_k, dropout = dropout)
    encoders = [EncoderLayerTF(embed_shape, d_k, d_ff, dropout = dropout, prefix = f'{name}_encoder_{i}', num_heads = num_heads) for i in range(num_layers)]

    # build model
    x = emebedding(inputs)
    for encoder in encoders:
        x = encoder(x)
    
    x = Flatten()(x)
    x = Dense(512, activation='gelu')(x)
    x = Dense(128, activation='gelu')(x)
    outputs = Dense(num_class, activation='softmax', name = f'{name}_output')(x)
    
    try:
        del outputs._keras_mask
    except AttributeError:
        pass
    
    model = Model(inputs, outputs, name = name)
    
    return model

# class BERTTFModel(Layer):
#     def __init__(self, num_layers, num_heads, seq_len, d_k, d_ff, vocab_size, num_class, dropout=0.1, name="BERTTF"):
#         super(BERTTFModel, self).__init__(name=name)

#         self.embed_shape = (seq_len, d_k)
#         self.embedding = BERTEmbeddingTF(seq_len, vocab_size, d_k, dropout=dropout)
#         self.encoders = [EncoderLayerTF(self.embed_shape, d_k, d_ff, dropout=dropout, prefix=f'{name}_encoder_{i}', num_heads=num_heads) for i in range(num_layers)]
        
#         self.flatten = Flatten()
#         self.dense1 = Dense(512, activation='gelu')
#         self.dropout1 = Dropout(0.2)
#         self.dense2 = Dense(128, activation='gelu')
#         self.output_layer = Dense(num_class, activation='softmax', name=f'{name}_output')

#     def call(self, inputs):
#         x = self.embedding(inputs)
#         for encoder in self.encoders:
#             x = encoder(x)
        
#         x = self.flatten(x)
#         x = self.dense1(x)
#         x = self.dropout1(x)
#         x = self.dense2(x)
#         outputs = self.output_layer(x)
#         return outputs
    
#     def get_config(self):
#         return super().get_config().update({"num_layers": self.num_layers, "num_heads": self.num_heads, "seq_len": self.seq_len, "d_k": self.d_k, "d_ff": self.d_ff, "vocab_size": self.vocab_size, "num_class": self.num_class, "dropout": self.dropout})
    
if __name__ == "__main__":
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # sentences = ["I love to explore new technologies. I love AI", "This is an example sentence.", "This is a question?", "This is an answer."]

    # input_id = []

    # for sentence in sentences:
    #     # Tokenize the sentences and generate segment labels with a maximum length of 15
    #     encoded_inputs = tokenizer.encode_plus(
    #         sentence, add_special_tokens=True, max_length=400, truncation=True, padding='max_length'
    #     )
    #     input_id.append(encoded_inputs['input_ids'])
        
    # # Convert input lists to NumPy arrays
    # input_id = np.array(input_id)
    
    seq_len = 20
    num_layers = 4
    num_heads = 8
    key_dim = 128
    ff_dim = 512
    dropout = 0.1
    num_class = 5
    
    model = BERTTF(num_layers, num_heads, seq_len, key_dim, ff_dim, 1000, num_class, dropout = dropout)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=star_mae)
    model.summary()