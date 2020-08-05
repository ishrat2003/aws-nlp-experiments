import tensorflow as tf
from .positionalEncoding import PositionalEncoding
from .encoder import Encoder

class SequenceEncoder(tf.keras.layers.Layer):
    
    def __init__(self, params, input_vocab_size, maximum_position_encoding):
        super(SequenceEncoder, self).__init__()
        self.params = params
        self.d_model = self.params.d_model
        num_heads = self.params.num_heads
        self.num_layers = self.params.num_layers
        dff = self.params.dff
        rate = self.params.dropout_rate
        '''
        Turns positive integrs into dense vector of fixed size.
        Model specification:
            input_dim = input_vocab_size (int > 0. Size of the vocabulary, i.e. maximum integer index + 1.)
            output_dim = d_model (Dimension of the dense embedding.)
            input_dim = input_vocab_size
            d_model = scaler (default 128)
        Input:
            2D tensor with shape: (batch_size, input_sequence_length).

        Output:
            3D tensor with shape: (batch_size, input_sequence_length, d_model).
        '''
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, self.d_model)
        
        '''
        Positional encoding:
        
        Input:
            maximum_position_encoding (= inputVocabSize)
            d_model = scaler (default 128)
            
        Output:
            self.positionalEmbedding (inputVocabSize x inputVocabSize x d_model)
        
        '''
        self.positionalEncoder = PositionalEncoding(maximum_position_encoding, self.d_model);
        self.positionalEmbedding = self.positionalEncoder.getEmbedding()

        self.encoderLayers = [Encoder(self.params, self.d_model, num_heads, dff, rate) for _ in range(self.num_layers)]

        '''
        Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.
        '''
        self.dropout = tf.keras.layers.Dropout(rate)
        return
        
    def call(self, x, training, mask):
        if (self.params.display_details == True) :
            print('----- SequenceEncoder -----')
            print('x', x.shape)
            print('mask', mask.shape)
            
        seqLenght = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.positionalEmbedding[:, :seqLenght, :] # (batch_size, input_seq_len, d_model)

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.encoderLayers[i](x, training, mask)
            
        if (self.params.display_details == True) :
            print('encoder output', x.shape)
            
        return x  # (batch_size, input_seq_len, d_model)