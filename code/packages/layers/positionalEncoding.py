import tensorflow as tf
import numpy as np

class PositionalEncoding:
    
    def __init__(self, totalPositions = 1000, dimensions = 512):
        self.totalPositions = totalPositions
        self.dimensions = dimensions
        return
    
    def getAngles(self, position, i):
        angleRates = 1 / np.power(10000, (2 * (i//2)) / np.float32(self.dimensions))
        return position * angleRates
    
    def getEmbedding(self):
        angleRates = self.getAngles(np.arange(self.totalPositions)[:, np.newaxis],
                          np.arange(self.dimensions )[np.newaxis, :])
  
        # apply sin to even indices in the array; 2i
        angleRates[:, 0::2] = np.sin(angleRates[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angleRates[:, 1::2] = np.cos(angleRates[:, 1::2])

        positionalEmbedding = angleRates[np.newaxis, ...]
        return tf.cast(positionalEmbedding, dtype=tf.float32)
    
    def pointWiseFeedForwardNetwork(self, d_model, dff):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
            ])