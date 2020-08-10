import tensorflow as tf
from layers.transformer import Transformer
from layers.mask import Mask
import matplotlib.pyplot as plt
import os
import datetime
import io

class Sequence():

    def __init__(self, params, sourceTokenizer, targetTokenizer, model = None):
        self.params = params
        self.maxLength = params['target_max_sequence_length']
        self.sourceTokenizer = sourceTokenizer
        self.targetTokenizer = targetTokenizer
        self.setModel(model)
        self.mask = Mask()
        self.source = None
        self.attentionWeights = None
        self.outputSentences = None
        self.plotDir = os.path.join(params['output_directory'], params['dataset_name'], datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.setTensorboard(self.plotDir)
        return
    
    def setModel(self, model = None):
        if model:
            self.model = model
        else:
            inputVocabSize = self.sourceTokenizer.vocab_size + 2
            targetVocabSize = self.targetTokenizer.vocab_size + 2
            self.model = Transformer(self.params, inputVocabSize, targetVocabSize, inputVocabSize, targetVocabSize)
        return
    
    def setTensorboard(self, logDir):
        self.writer = tf.summary.create_file_writer(logDir)
        return

    def process(self, source):
        self.source = source
        startToken = [self.sourceTokenizer.vocab_size]
        endToken = [self.sourceTokenizer.vocab_size + 1]

        sourceInput = startToken + self.sourceTokenizer.encode(source) + endToken
        encoderInput = tf.expand_dims(sourceInput, 0)

        # as the target is english, the first word to the transformer should be the
        # english start token.
        decoderInput = [self.targetTokenizer.vocab_size]
        target = tf.expand_dims(decoderInput, 0)

        for i in range(self.maxLength):
            encoderPaddingMask, decoderTargetPaddingAndLookAheadMask, decoderPaddingMask = self.mask.createMasks(encoderInput, target)
        
            if (self.params['display_details'] == True) :
                print('encoderInput: ', encoderInput.shape)
                print('target: ', target.shape)
                print('encoderPaddingMask: ', encoderPaddingMask.shape)
                print('decoderTargetPaddingAndLookAheadMask: ', decoderTargetPaddingAndLookAheadMask.shape)
                print('decoderPaddingMask: ', decoderPaddingMask.shape)
            # predictions.shape == (batch_size, seq_len, vocab_size)    
            predictions, attentionWeights = self.model(encoderInput, 
                target,
                False,
                encoderPaddingMask,
                decoderTargetPaddingAndLookAheadMask,
                decoderPaddingMask)
            
            # select the last word from the seq_len dimension
            predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

            
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if predicted_id == self.targetTokenizer.vocab_size + 1:
                return tf.squeeze(target, axis=0)

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            target = tf.concat([target, predicted_id], axis=-1)

        self.output = tf.squeeze(target, axis=0)
        self.outputSentence = self.targetTokenizer.decode([i for i in self.output if i < self.targetTokenizer.vocab_size])
        self.attentionWeights = attentionWeights
        
        return self.outputSentence
    
    def plotAttentions(self, layer = 0):
        if not self.attentionWeights:
            return
        
        for layer in self.attentionWeights.keys():
            self.plotAttention(layer)
            
        return
    
    
    def plotAttention(self, layer = 0):
        if not self.attentionWeights or (layer not in self.attentionWeights.keys()):
            return

        fig = plt.figure(figsize=(16, 8))
        sentence = self.sourceTokenizer.encode(self.source)
        attention = tf.squeeze(self.attentionWeights[layer], axis=0)

        for head in range(self.attentionWeights.shape[0]):
            ax = fig.add_subplot(2, 4, head+1)

            # plot the attention weights
            ax.matshow(self.attentionWeights[head][:-1, :], cmap='viridis')

            fontdict = {'fontsize': 10}

            ax.set_xticks(range(len(sentence)+2))
            ax.set_yticks(range(len(outputSentence)))

            ax.set_ylim(len(outputSentence)-1.5, -0.5)

            ax.set_xticklabels(
            ['<start>']+[self.sourceTokenizer.decode([i]) for i in sentence]+['<end>'], 
            fontdict=fontdict, rotation=90)

            ax.set_yticklabels([self.targetTokenizer.decode([i]) for i in self.output
                            if i < self.targetTokenizer.vocab_size], 
                            fontdict=fontdict)

            ax.set_xlabel('Head {}'.format(head+1))

        plt.tight_layout()
        #plt.show()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        
        with self.writer.as_default():
            tf.summary.image("Attention weight", image, step=layer)
            self.writer.flush()
        return