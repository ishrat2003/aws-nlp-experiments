import tensorflow as tf
import numpy as np

class ScaledDot:
    
    def __init__(self, params):
        self.params = params
        return
    
    def attention(self, q, k, v, mask):
        """
        Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead) 
        but it must be broadcastable for addition.

        Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable 
        to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
        output, attention_weights
        """
        if (self.params.display_details == True) :
            print('Scaled dot q :', q.shape)
            print('Scaled dot k: ', k.shape)
            print('Scaled dot v: ', v.shape)
            print('Scaled dot mask: ', mask.shape)
            
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        if (self.params.display_details == True) :
            print('Scaled dot matmul_qk :', matmul_qk.shape)
            
        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        if (self.params.display_details == True) :
            print('Scaled dot dk :', dk)

        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if (self.params.display_details == True) :
            print('Scaled dot scaled_attention_logits :', scaled_attention_logits.shape)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
            
        if (self.params.display_details == True) :
            print('Scaled dot scaled_attention_logits :', scaled_attention_logits.shape)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
        if (self.params.display_details == True) :
            print('Scaled dot attention_weights :', attention_weights.shape)
            
        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
        
        if (self.params.display_details == True) :
            print('Scaled dot output :', output.shape)
            
        return output, attention_weights
