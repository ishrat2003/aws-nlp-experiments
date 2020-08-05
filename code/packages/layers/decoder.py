import tensorflow as tf
from .multiHead import MultiHead as MultiHeadAttention
from .feedForward import FeedForward

class Decoder(tf.keras.layers.Layer):
    
    def __init__(self, params, d_model, num_heads, dff, rate=0.1):
        super(Decoder, self).__init__()
        self.params = params
        self.ffDecoder = FeedForward()
        
        self.mha1 = MultiHeadAttention(self.params, d_model, num_heads)
        self.mha2 = MultiHeadAttention(self.params, d_model, num_heads)

        self.ffn = self.ffDecoder.pointWiseFeedForwardNetwork(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        return

    def call(self, x, encoderOutput, training, lookAheadMask, paddingMask):
        if (self.params.display_details == True) :
            print('----- Decoder -----')
            print('x', x.shape)
            print('encoderOutput', encoderOutput.shape)
            print('lookAheadMask', lookAheadMask.shape)
            print('paddingMask', paddingMask.shape)
            
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attnWeightsBlock1 = self.mha1(x, x, x, lookAheadMask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attnWeightsBlock2 = self.mha2(encoderOutput, encoderOutput, out1, paddingMask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffnOutput = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffnOutput = self.dropout3(ffnOutput, training=training)
        out3 = self.layernorm3(ffnOutput + out2)  # (batch_size, target_seq_len, d_model)

        if (self.params.display_details == True) :
            print('out1', out1.shape)
            print('out2', out2.shape)
            print('out3', out3.shape)
            print('attnWeightsBlock1', attnWeightsBlock1.shape)
            print('attnWeightsBlock2', attnWeightsBlock2.shape)
            
        return out3, attnWeightsBlock1, attnWeightsBlock2