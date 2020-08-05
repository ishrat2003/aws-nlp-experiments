import tensorflow as tf
from .multiHead import MultiHead as MultiHeadAttention
from .feedForward import FeedForward

class Encoder(tf.keras.layers.Layer):
    
    def __init__(self, params, d_model, num_heads, dff, rate=0.1):
        super(Encoder, self).__init__()
        self.params = params
        self.ffEncoder = FeedForward()
        
        self.mha = MultiHeadAttention(self.params, d_model, num_heads)
        self.ffn = self.ffEncoder.pointWiseFeedForwardNetwork(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        return

    def call(self, x, training, mask):
        if (self.params.display_details == True) :
            print('----- Encoder -----')
            print('x', x.shape)
            print('mask', mask.shape)
            
        attnOutput, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attnOutput = self.dropout1(attnOutput, training=training)
        out1 = self.layernorm1(x + attnOutput)  # (batch_size, input_seq_len, d_model)

        ffnOutput = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffnOutput = self.dropout2(ffnOutput, training=training)
        out2 = self.layernorm2(out1 + ffnOutput)  # (batch_size, input_seq_len, d_model)
        
        if (self.params.display_details == True) :
            print('attnOutput', attnOutput.shape)
            print('out1', out1.shape)
            print('ffnOutput', ffnOutput.shape)
            print('out2', out2.shape)

        return out2