import tensorflow as tf
import numpy as np
from .scaledDot import ScaledDot

class MultiHead(tf.keras.layers.Layer):
    
    def __init__(self, params, d_model, num_heads):
        super(MultiHead, self).__init__()
        self.params = params
        
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)
        self.scaledDotProduct = ScaledDot(self.params)
        return

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        xTranspose = tf.transpose(x, perm=[0, 2, 1, 3])
        
        if (self.params.display_details == True) :
            print('splitting head: ', batch_size, -1, self.num_heads, self.depth)
            print('after reshape x:', x.shape)
            print('after transpose [0, 2, 1, 3] x:', xTranspose.shape)
        return xTranspose

    def call(self, v, k, q, mask):
        if (self.params.display_details == True) :
            print('----- Multi Head -----')
            print('q - input: ', q.shape)
            print('k - input: ', k.shape)
            print('v - input: ', v.shape)
            
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaledDotProduct.attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                    (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        if (self.params.display_details == True) :
            print('----- Multi Head -----')
            print('batch_size: ', batch_size)
            print('wq: ', q.shape)
            print('wk: ', k.shape)
            print('wv: ', v.shape)
            print('q: ', q.shape)
            print('k: ', k.shape)
            print('v: ', v.shape)
            print('scaled_attention: ', scaled_attention.shape)
            print('attention_weights: ', attention_weights.shape)
            print('concat_attention: ', concat_attention.shape)
            print('output: ', output.shape)

        return output, attention_weights
    