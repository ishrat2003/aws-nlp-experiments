import tensorflow as tf
from layers.transformer import Transformer
from layers.customScheduler import CustomScheduler
from layers.mask import Mask
from layers.loss import Loss

import time
import numpy
import  sys
import os
import datetime

class Trainer:
    
    def __init__(self, params, tokenizerSource, tokenizerTarget):
        self.params = params
        self.dModel = self.params['d_model']
        self.optimizer = self.getOptimizer()
        self.writeAfterNBatches = 50
        self.model = None
        self.mask = Mask()
        self.loss = Loss()
        self.setLossMetrics()
        self.setAccuracyMetrics()
        self.predictor = None
        self.evaluator = None
        self.sourceTokenizer = tokenizerSource
        self.targetTokenizer = tokenizerTarget
        return
    
    def setTensorboard(self, logDir):
        logDir = os.path.join(logDir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        # Display with the tensorflow file writer
        self.writer = tf.summary.create_file_writer(logDir)
        return
    
    def getOptimizer(self):
        learningRate = CustomScheduler(self.dModel)
        return tf.optimizers.Adam(learningRate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    def setLossMetrics(self):
        self.trainLoss = tf.keras.metrics.Mean(name='train_loss')
        return
    
    def setAccuracyMetrics(self):
        self.trainAccuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        return
    
    def setValidationDataset(self, dataset):
        self.validationDataset = dataset
        return

    def setModel(self, model):
        self.model = model
        return
    
    def setPredictor(self, predictor):
        self.predictor = predictor
        return
    
    def setEvaluator(self, evaluator):
        self.evaluator = evaluator
        return
    
    def setCheckpoint(self, checkpointPath):
        self.ckpt = tf.train.Checkpoint(transformer=self.model, optimizer=self.optimizer)
        self.ckptManager = tf.train.CheckpointManager(self.ckpt, checkpointPath, max_to_keep=3)

        # if a checkpoint exists, restore the latest checkpoint. https://www.tensorflow.org/guide/checkpoint#loading_mechanics
        if self.ckptManager.latest_checkpoint:
            status = self.ckpt.restore(self.ckptManager.latest_checkpoint)
            status.assert_consumed()
            print ('Latest checkpoint restored!!')
            
        return
    
    def process(self, epochs, dataset):
        # The @tf.function trace-compiles train_step into a TF graph for faster
        # execution. The function specializes to the precise shape of the argument
        # tensors. To avoid re-tracing due to the variable sequence lengths or variable
        # batch sizes (the last batch is smaller), use input_signature to specify
        # more generic shapes.

        trainStepSignature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        ]
        # @tf.function(input_signature=trainStepSignature)
        def trainStep(source, target):
            # tf print doesn't work in jupyter
            # tf.print(source, output_stream=sys.stdout)
            targetInput = target[:, :-1]
            targetReal = target[:, 1:]

            encoderPaddingMask, decoderTargetPaddingAndLookAheadMask, decoderPaddingMask = self.mask.createMasks(source, targetInput)

            if (self.params['display_details'] == True) :
                print('trainStep target: ', target.shape)
                print('trainStep targetInput: ', targetInput.shape)
                print('trainStep targetReal: ', targetReal.shape)
                print('trainStep encoderPaddingMask', encoderPaddingMask.shape)
                print('trainStep decoderTargetPaddingAndLookAheadMask', decoderTargetPaddingAndLookAheadMask.shape)
                print('trainStep decoderPaddingMask', decoderPaddingMask.shape)
                
            with tf.GradientTape() as tape:
                if (self.params['display_details'] == True) :
                    print('G tape, source', source.shape, tf.shape(source))
                    print('G tape, targetInput', targetInput.shape, tf.shape(targetInput))
                predictions, _ = self.model(source, 
                    targetInput, 
                    True, 
                    encoderPaddingMask, 
                    decoderTargetPaddingAndLookAheadMask, 
                    decoderPaddingMask)
                    
                loss = self.loss.lossFunction(targetReal, predictions)

            gradients = tape.gradient(loss, self.model.trainable_variables)    
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            self.trainLoss(loss)
            self.trainAccuracy(targetReal, predictions)
            return

        for epoch in range(epochs):
            self.startEpoch(epoch)
            batch = 0
            # inp -> portuguese, tar -> english
            for (batch, (source, target)) in enumerate(dataset):
                if (self.params['display_details']) :
                    print('Batch: ', batch)
                    print('source', source.shape)
                    print('target', target.shape)
                trainStep(source, target)
                self.endBatch(batch, epoch)

            self.endEpoch(batch, epoch)
                
        return
    
    def evaluation(self, batch, epoch):
        if not self.validationDataset:
            return

        self.predictor.setModel(self.model)
        for (batch, (source, target)) in enumerate(self.validationDataset):
                generated = self.predictor.process(source)
                targetToCompare = self.targetTokenizer.decode([i for i in target if i < self.targetTokenizer.vocab_size])
                print('e target', targetToCompare)
                print('e generate', generated)
                score = self.evaluator.getScore(targetToCompare, generated)
                
                if (self.params['display_details'] == True) :
                    print('Batch: ', batch)
                    print('source: ', source)
                    print('target: ', target)
                    print('generated: ', generated)
                    print('score: ', score)
                
                self.saveSummary(batch, epoch, score)
                
        return
    
    def startEpoch(self, epoch):
        print('Starting epoch: ', (epoch+1))
        self.start = time.time()
        self.trainLoss.reset_states()
        self.trainAccuracy.reset_states()
        return
    
    def endBatch(self, batch, epoch):
        if batch % self.writeAfterNBatches == 0:
            print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f} Total {}'.format(epoch + 1, batch, self.trainLoss.result(), self.trainAccuracy.result(), self.getTotalProcessed(batch)))
            self.saveCheckPoint(batch, epoch);
            self.saveSummary(batch, epoch)
        return

    def endEpoch(self, batch, epoch):
        self.saveCheckPoint(batch, epoch)
        self.evaluation(batch, epoch)
        
        print ('Epoch {} Loss {:.4f} Accuracy {:.4f} Total {}'.format(epoch + 1, self.trainLoss.result(), self.trainAccuracy.result(), self.getTotalProcessed(batch)))
        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - self.start))
        return
    
    def getTotalProcessed(self, batch):
        return (batch + 1) * self.params['batch_size']
    
        
    def saveSummary(self, batch, epoch, score = None):
        with self.writer.as_default():
            tf.summary.scalar("batch-loss", self.trainLoss.result(), step=batch)
            tf.summary.scalar("batch-accuracy", self.trainAccuracy.result(), step=batch)
            
            if score:
                tf.summary.scalar("epoch-loss", self.trainLoss.result(), step=epoch)
                tf.summary.scalar("epoch-accuracy", self.trainAccuracy.result(), step=epoch)
                for key in score.keys():
                    for matrixKey in score[key]:
                        trackingKey = str('epoch-' + key + '-' + matrixKey)
                        tf.summary.scalar(trackingKey, score[key][matrixKey], step=epoch)
        
        self.writer.flush()
        return
    
    def saveCheckPoint(self, batch, epoch):
        ckptSavePath = self.ckptManager.save()
        print ('Saving checkpoint for epoch {} batch {} at {}'.format(epoch+1, batch, ckptSavePath))
        return
