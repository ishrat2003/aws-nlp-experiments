import tensorflow as tf

class Loss:    

    def __init__(self):
        self.lossObject = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        return

    def lossFunction(self, real, prediction):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.lossObject(real, prediction)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)