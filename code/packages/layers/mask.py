import tensorflow as tf
import numpy as np

class Mask:

    def createPaddingMask(self, sequence):
        '''
        Mask all the pad tokens in the batch of sequence. 
        It ensures that the model does not treat padding as the input. 
        The mask indicates where pad value 0 is present: it outputs a 1 
        at those locations, and a 0 otherwise.
        Example:
        input:
        sequence = [[7, 6, 0, 0, 1], 
            [1, 2, 3, 0, 0], 
            [0, 0, 0, 4, 5]]
            
        process:
        tf.math.equal(sequence, 0) =
            [[False False  True  True False]
            [False False False  True  True]
            [ True  True  True False False]]
            
        tf.cast(tf.math.equal(sequence, 0), tf.float32) =
            [[0. 0. 1. 1. 0.]
            [0. 0. 0. 1. 1.]
            [1. 1. 1. 0. 0.]]
            
        sequence[:, tf.newaxis, tf.newaxis, :] = 
        [
            [[
                [0. 0. 1. 1. 0.]
            ]]
            [[
                [0. 0. 0. 1. 1.]
            ]]
            [[
                [1. 1. 1. 0. 0.]
            ]]
        ]
        1st dimension = batch size
        2nd dimension = 1
        3rd dimension = 1
        4th dimension = sequence length
        output:
        
        '''
        sequence = tf.cast(tf.math.equal(sequence, 0), tf.float32)
        # add extra dimensions to add the padding
        # to the attention logits.
        return sequence[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
    
    def createLookAheadMask(self, size):
        '''
        The look-ahead mask is used to mask the future tokens in a sequence. 
        In other words, the mask indicates which entries should not be used.
        This means that to predict the third word, only the first and second word 
        will be used. Similarly to predict the fourth word, only the first, second 
        and the third word will be used and so on.
        Exmple:
        input:
        size = 4
        output:
        [[0. 1. 1. 1.]
        [0. 0. 1. 1.]
        [0. 0. 0. 1.]
        [0. 0. 0. 0.]]
        '''
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (size, size)
    
    def createMasks(self, source, target):
        # Encoder padding mask
        encoderPaddingMask = self.createPaddingMask(source)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        decoderPaddingMask = self.createPaddingMask(source)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by 
        # the decoder.
        lookAheadMask = self.createLookAheadMask(tf.shape(target)[1])
        decoderTargetPaddingMask = self.createPaddingMask(target)
        decoderTargetPaddingAndLookAheadMask = tf.maximum(lookAheadMask, decoderTargetPaddingMask)
        return encoderPaddingMask, lookAheadMask, decoderPaddingMask