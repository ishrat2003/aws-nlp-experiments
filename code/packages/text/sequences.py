import os
import tensorflow as tf

class Sequences:
    
    def __init__(self, params, tokenizerSource, tokenizerTarget, dataProcessor):
        self.params = params
        self.tokenizerSource = tokenizerSource
        self.tokenizerTarget = tokenizerTarget
        self.dataProcessor = dataProcessor
        return
    
    def getTokenizedDataset(self, dataset, validation = False):
        def encode(data):
            source = [self.tokenizerSource.vocab_size] + self.tokenizerSource.encode(self.dataProcessor.getText(data, False, True, False)) + [self.tokenizerSource.vocab_size + 1]           
            target = [self.tokenizerTarget.vocab_size] + self.tokenizerTarget.encode(self.dataProcessor.getAbstractText(data, True, False)) + [self.tokenizerTarget.vocab_size + 1]          
            return source, target
        
        def tfEncode(data, label):
            source, target = tf.py_function(encode, [data], [tf.int64, tf.int64])
            source.set_shape([None])
            target.set_shape([None])
            print('tfEncode source', source)
            print('tfEncode target', target)
            return source, target
        
        # def filterMaxLength(source, target):
        #     return tf.logical_and(tf.size(source) <= self.params['source_max_sequence_length'], tf.size(target) <= self.params['target_max_sequence_length'])

        tokenizedDataset = dataset.map(tfEncode)
        # tokenizedDataset = tokenizedDataset.filter(filterMaxLength)
        
        if not validation:
            # cache the dataset to memory to get a speedup while reading from it.
            tokenizedDataset = tokenizedDataset.cache()
            if (self.params['total_items'] == 0):
                # No shuffle is only n items of the dataset is processed. Otherwise, training doesn;'t get good accuracy
                tokenizedDataset = tokenizedDataset.shuffle(self.params['buffer_size'])
            tokenizedDataset = tokenizedDataset.padded_batch(self.params['batch_size'], padded_shapes=([None],[None]))
            tokenizedDataset = tokenizedDataset.prefetch(tf.data.experimental.AUTOTUNE)
            
        return tokenizedDataset
    
    def printSample(self, dataset):
        print('Printing sample sequences')
        for (source, target) in dataset:
            
            #ßsource, target = next(iter(dataset))
            print('----------------------')
            print('Source')
            print('----------------------')
            print(source)
            print('----------------------')
            print('Target')
            print('----------------------')
            print(target)
            return
        return
    
       
