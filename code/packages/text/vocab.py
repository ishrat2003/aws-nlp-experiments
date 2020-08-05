import os, logging
import tensorflow_datasets as tfds

class Vocab:

    def __init__(self, dataset, vocab_size=2**16):
        self.vocabPath = dataset.getVocabPath()
        self.dataset = dataset
        self.vocabSize = vocab_size
        return

    def get(self, data, type = 'source'):
        vocabPath = self.getVocabPath(type)
        print('Vocab path: ' + vocabPath + '.subwords')
        if os.path.exists(vocabPath + '.subwords'):
            logging.info("# Loading vocab")
            return tfds.features.text.SubwordTextEncoder.load_from_file(vocabPath)
    
        logging.info("# Building vocab")
        return self.build(data, type)
    
    def build(self, data, type = 'source'):
        encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(self.dataset.getGenerator(data, type), self.vocabSize)
        encoder.save_to_file(self.getVocabPath(type))
        return encoder
    
    def getVocabPath(self, type = 'source'):
        return self.vocabPath + '_' + type
    
    def printSample(self, tokenizer, text):
        print('Vocab size: {}'.format(tokenizer.vocab_size))
        print('Input string: {}'.format(text))
        
        tokenizedString = tokenizer.encode(text)
        print('Tokenized string: {}'.format(tokenizedString))

        originalString = tokenizer.decode(tokenizedString)
        print('Decoded string: {}'.format(originalString))

        for token in tokenizedString:
            print('{} ----> {}'.format(token, tokenizer.decode([token])))
            
        return