from __future__ import absolute_import, division, print_function, unicode_literals
import os
import json
import pickle
import sys
import traceback
import logging
import pandas as pd
#import sagemaker
from os import listdir
from os.path import isfile, join

# Registering local packages
import sys
packagesPath = os.path.join(os.getcwd(), "packages")
sys.path.append(packagesPath)


# These are the paths to where SageMaker mounts interesting things in your container.

prefix = '/opt/ml/'

input_path = prefix + 'input/data'
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')
params_path = os.path.join(prefix, 'input/config/hyperparameters.json')

logging.basicConfig(level=logging.INFO) 

print(sys.argv);
print(os.getenv('env_var_name'))

# The function to execute the training.
def train():
    print('Starting the training123.' + '\n')
    try:
        print('Current working directory: ' + os.getcwd() + '\n')
        logging.info("# 1. Loading script params ")
        logging.info("# ================================")
        
        f = open(params_path, 'r')
        file_contents = f.read()
        print (file_contents)
        f.close()
        
        logging.info("# ================================")
        
        print(listdir(os.getcwd()))
        print(listdir('/opt/ml/input/'))
        print(listdir('/opt/ml/input/data/'))
        print(listdir('/opt/ml/input/data/training/'))
        logging.info("# ================================")
        with open(os.path.join(output_path, 'sample_output'), 'w') as s:
            s.write('training out')
            
        logging.info('Training complete.11111????')
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
            
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)

if __name__ == '__main__':
    train()

    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)

