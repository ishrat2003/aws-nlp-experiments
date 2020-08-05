import tensorflow as tf
from .positionalEncoding import PositionalEncoding
from .decoder import Decoder

class SequenceDecoder(tf.keras.layers.Layer):
    
    def __init__(self, params, target_vocab_size, maximum_position_encoding):
        self.params = params
        self.num_layers = params.num_layers
        self.d_model = params.d_model
        num_heads = params.num_heads
        dff = params.dff
        rate = params.dropout_rate
        
        super(SequenceDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, self.d_model)
        self.positionalEncoder = PositionalEncoding(maximum_position_encoding, self.d_model);
        self.positionalEmbedding = self.positionalEncoder.getEmbedding()

        self.decoderLayers = [Decoder(self.params, self.d_model, num_heads, dff, rate) for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        return

    def call(self, x, encoderOutput, training, lookAheadMask, paddingMask):
        if (self.params.display_details == True) :
            print('----- SequenceDecoder -----')
            print('x', x.shape)
            print('encoderOutput', encoderOutput.shape)
            print('lookAheadMask', lookAheadMask.shape)
            print('paddingMask', paddingMask.shape)
    
        seq_len = tf.shape(x)[1]
        attentionWeights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.positionalEmbedding[:, :seq_len, :]

        if (self.params.display_details == True) :
            print('output after positional encoding', x.shape)
            
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.decoderLayers[i](x, encoderOutput, training, lookAheadMask, paddingMask)
            attentionWeights['decoder_layer{}_block1'.format(i+1)] = block1
            attentionWeights['decoder_layer{}_block2'.format(i+1)] = block2

        if (self.params.display_details == True) :
            print('output after decoder layers', x.shape)
            print('attentionWeights1 after decoder layers', attentionWeights['decoder_layer1_block1'].shape)
            print('attentionWeights2 after decoder layers', attentionWeights['decoder_layer1_block2'].shape)

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attentionWeights