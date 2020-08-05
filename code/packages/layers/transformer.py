import tensorflow as tf
from .sequenceEncoder import SequenceEncoder
from .sequenceDecoder import SequenceDecoder

class Transformer(tf.keras.Model):
    
    def __init__(self, params, input_vocab_size, target_vocab_size, pe_input, pe_target):
        super(Transformer, self).__init__()
        self.params = params
        num_layers = params.num_layers
        d_model = params.d_model
        num_heads = params.num_heads
        dff = params.dff
        rate = params.dropout_rate
        '''
        Encoder:
        Combination of layers dedicated to train source sequence.
        Input:
            source 
                (dataset_size x source_sequence_length)
            encoderPaddingMask 
                (dataset_size x 1 x 1 x source_sequence_length) 
                (value 1 when corresponding source value = 0)
                (value 0 when corresponding source value != 0)
        Output:
            encoderOutput
                (dataset_size x source_sequence_length)
        '''
        self.encoder = SequenceEncoder(self.params, input_vocab_size, pe_input)
        
        '''
        Decoder:
        Combination of layers dedicated to train traget sequence.
        Input:
        target
            (dataset_size x target_sequence_length)
        encoderOutput
        decoderTargetPaddingAndLookAheadMask
            (dataset_size x 1 x 1 x target_sequence_length) 
            (value 1 when corresponding target value = 0)
            (value 1 when corresponding target value is a future/look ahead value)
            (value 0 otherwise)
        decoderPaddingMask
            (dataset_size x 1 x 1 x target_sequence_length) 
            (value 1 when corresponding target value = 0)
            (value 0 when corresponding target value != 0)
        
        Output:
            decoderOutput (batch_size, target_seq_len, d_model)
            attentionWeights (batch_size, target_seq_len, d_model)
        '''
        self.decoder = SequenceDecoder(self.params, target_vocab_size, pe_target)
        
        '''
        Final layer:
        Dense layer implements
        output = activation(dot(input, weight) + bias)
        bias and weight matrices are created and maintained by the layer.
        bias is set if use_bias = true passed by the function.
        activation function can be specified, if not specified than it is 'linear' by default.
        units is the first required param that specifies the dimensionality of the output space.
        units = target_vocab_size for the final layer.
        
        Input: 
            decoderOutput
        Output: 
            outputTensor (batch_size x target_vocab_size)
        '''
        self.finalLayer = tf.keras.layers.Dense(target_vocab_size)
    
        if self.params.display_network == True:
            print('Transformer (3 layers)')
            print('SequenceEncoder layer')
            print('-- Input: source, encoderPaddingMask')
            print('-- Output: encoderOutput')
            print('SequenceDecoder layer')
            print('-- Input: target, encoderOutput, decoderTargetPaddingAndLookAheadMask, decoderPaddingMask')
            print('-- Output: decoderOutput, attentionWeights')
            print('Final Dense layer')
            print('-- Input: decoderOutput')
            print('-- Output: finalOutput')
        return

    def call(self, source, target, training, encoderPaddingMask, decoderTargetPaddingAndLookAheadMask, decoderPaddingMask):
        if (self.params.display_details == True) :
            print('=========== Transformer ===========')
            
        encoderOutput = self.encoder(source, training, encoderPaddingMask)  # (batch_size, inp_seq_len, d_model)
        if (self.params.display_details == True) :
            print('transformer encoderOutput:', encoderOutput.shape)
            
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        decoderOutput, attentionWeights = self.decoder(target, encoderOutput, training, decoderTargetPaddingAndLookAheadMask, decoderPaddingMask)
        if (self.params.display_details == True) :
            print('transformer decoderOutput:', decoderOutput.shape)
            print('transformer attentionWeights:', attentionWeights)

        finalOutput = self.finalLayer(decoderOutput)  # (batch_size, tar_seq_len, target_vocab_size)
        if (self.params.display_details == True) :
            print('transformer finalOutput:', finalOutput.shape)
        
        return finalOutput, attentionWeights