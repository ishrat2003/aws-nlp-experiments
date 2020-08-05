from text.vocab import Vocab
import os, logging
import tensorflow_datasets as tfds

class LCVocab(Vocab):

    def __init__(self, dataset, params = {}, vocab_size=2**16):
        super().__init__(dataset, vocab_size)
        self.params = params
        print(self.params)
        self.prefix = 'contextual_' + self.params['mode'] + '_'
        self.vocabPath = dataset.getVocabPath(self.prefix )
        return

