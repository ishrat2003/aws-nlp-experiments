from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import traceback
import logging

# Registering local packages
import sys
packagesPath = os.path.join(os.getcwd(), "packages")
sys.path.append(packagesPath)

from params import Input as InputParams
from params import Output as OutputParams
from utility import Timer
from dataset.core import Core as Dataset
from contextualVocab.lcVocab import LCVocab
from text.sequences import Sequences

paramsProcessor = InputParams("/opt/ml/input/config/hyperparameters.json")
params = paramsProcessor.getAll()
outputProcessor = OutputParams()
outputProcessor.addInfo('hyperparameters', params)
        
outputPath = params['output_directory'] if params['output_directory'] else "/opt/ml/output";
logPath = os.path.join(params['output_directory'], "process.log")
logging.basicConfig(filename=logPath,level=logging.DEBUG)

def train():
    logging.info('Starting the training.')
    try:
        # ========== Setup =============
        logging.info('Current working directory: ' + os.getcwd())
        Timer.start('training')
        logging.info("# Starting ")
        logging.info("# ================================")
        # ========== Loading dataset processor =============
        logging.info("# 2.1. Loading raw dataset (" + str(params['dataset_percentage']) + "%)")
        print('params', params['data_directory'])
        dataset = Dataset(params['dataset_name'], params['data_directory'], params)
        dataProcessor = dataset.get(float(params['dataset_percentage']), int(params['total_items']))
        trainingSet, validationSet = dataProcessor.get()
        if not trainingSet or not validationSet:
            logging.error('No dataset found')
            sys.exit(1)
            
        logging.info("# 2.2. Preparing vocab (source + target)")
        vocabProcessor = LCVocab(dataProcessor, params)
        logging.info("# Preparing vocab (source)")
        tokenizerSource = vocabProcessor.get(trainingSet, 'source')
        logging.info("# Preparing vocab (target)")
        tokenizerTarget = vocabProcessor.get(trainingSet, 'target')
        logging.info("# Sample output using source vocab")
        sampleString = 'This news is great'
        vocabProcessor.printSample(tokenizerSource, sampleString)
        logging.info("# Sample output using target vocab")
        vocabProcessor.printSample(tokenizerTarget, sampleString)

        # ============= End ===============
        logging.info("# Finishing")
        logging.info("# ================================")
        Timer.stop('training')
        timers = Timer.getFormattedTimers()
        outputProcessor.addInfo('timers', timers)
        logging.info(timers)
        dataProcessor.saveInfo()
        outputProcessor.saveInfo(os.path.join(outputPath, 'info.json')) 
        logging.info('Training complete')
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(outputPath, 'failure'), 'w+') as failedFile:
            failedFile.write('Exception during training: ' + str(e) + '\n' + trc)
            
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(1)

if __name__ == '__main__':
    train()

    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)

