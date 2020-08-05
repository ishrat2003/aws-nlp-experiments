import pathlib
import tensorflow as tf
import json
import lc
import os
import re
from .base import Base

class BaseCsv(Base):
    
    def get(self):
        csvPath = self.__getCsvPath()
        dataset = tf.data.experimental.make_csv_dataset(csvPath, batch_size=1, select_columns = ['name', 'content'], label_name='name', shuffle=False)
        return dataset
    
    def getTrainingSet(self):
        dataset = self.get()
        if self.totalItems:
            return dataset.take(self.totalItems)
        return dataset
    
    def getText(self, rawData):
        (source, label) = rawData
        text = re.compile(r'<[^>]+>').sub('', source['content'].numpy()[0].decode('utf-8'))
        text = re.compile(r'&[a-z0-9]+;').sub(' ', text)
        return text

    def getTitle(self, rawData):
        return self.getLabel(rawData)
    
    def getLabel(self, rawData):
        (source, label) = rawData
        return label.numpy()[0].decode('utf-8')

    def __getCsvPath(self):
        return os.path.join(self.path, 'training.csv')

