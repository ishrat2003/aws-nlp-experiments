from evaluation.rouge import Rouge
import os
from file.core import Core as File

from datetime import datetime
from .peripheral import Peripheral
class BasicEvaluate:

    def __init__(self, dataset, params):
        self.dataset = dataset
        self.params = params
        self.allowedTypes = ['NN', 'NNP', 'NNS', 'NNPS']
        self.rouge = Rouge( self.params, ['rouge1'])
        self.topScorePrecentages = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.positionContributingFactor = params.position_contributing_factor
        self.occuranceContributingFactor = params.occurance_contributing_factor
        self.file = self.getFile()
        self.fileSummary = self.getFile('summary')
        self.setDefaultPOSGroups()
        return
    
    def setDefaultPOSGroups(self):
        self.posGroups = {}
        self.posGroups['n'] = ['NN', 'NNP', 'NNS', 'NNPS']
        return
    
    def setAllPOSGroups(self):
        self.posGroups = {}
        self.posGroups['n'] = ['NN', 'NNP', 'NNS', 'NNPS']
        self.posGroups['adj'] = ['JJ', 'JJR', 'JJS']
        self.posGroups['nAdj'] = ['NN', 'NNP', 'NNS', 'NNPS', 
                                  'JJ', 'JJR', 'JJS']
        self.posGroups['v'] = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        self.posGroups['adv'] = ['RB', 'RBR', 'RBS']
        self.posGroups['vAdv'] = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                                  'RB', 'RBR', 'RBS']
        self.posGroups['nAdjAdvV'] = ['NN', 'NNP', 'NNS', 'NNPS', 
                                      'JJ', 'JJR', 'JJS',
                                      'RB', 'RBR', 'RBS', 
                                      'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        self.posGroups['all'] = ["CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD",
                                 "NN","NNS","NNP","NNPS","PDT","POS","PRP","PRP$","RB","RBR",
                                 "RBS","RP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ",
                                 "WDT","WP","WP$","WRB"]
        return
        
    def getFile(self, prefix = ''):
        now = datetime.now()
        dateString = now.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.params.data_directory, 'cwr', self.getFileName(prefix + '_' + str(dateString)))
        file = File(path)
        return file
    
    def getFileName(self, prefix = ''):
        return self.params.dataset_name + '_' + prefix + '_pos' + str(self.positionContributingFactor) + '_occ' + str(self.occuranceContributingFactor) + '.csv';

    def setAllowedTypes(self, allowedTypes):
        self.allowedTypes = allowedTypes
        return

    def process(self):
        self.initInfo()

        data = self.dataset.getTrainingSet()

        for item in data:
            row = self.processItem(0, item)
            self.file.write(row)
            print(self.info['total'])
            self.info['total'] += 1

        self.summarizeInfo()
        return
    
 
    def processItem(self, batch, sourceText, targetText):
        return {}
    
    def evaluate(self, generatedContributor, expectedContributor, posType, topScorePercentage):
        if self.params.display_details:
            print('expectedContributor::: ', expectedContributor)
            print('generatedContributor::: ', generatedContributor)
        row = {}
        evaluationScore = self.rouge.getScore(expectedContributor, generatedContributor)      
        suffix = self.getSuffix(posType, topScorePercentage)
        row['expected'] = expectedContributor
        row['generated_contributor_' + suffix] = generatedContributor
        row['rouge1_precision_' + suffix] = evaluationScore['rouge1']['precision']
        row['rouge1_recall_' + suffix] = evaluationScore['rouge1']['recall']
        row['rouge1_fmeasure_' + suffix] = evaluationScore['rouge1']['fmeasure']
        
        self.info['total_precision'][suffix] += evaluationScore['rouge1']['precision']
        self.info['total_recall'][suffix] += evaluationScore['rouge1']['recall']
        self.info['total_fmeasure'][suffix] += evaluationScore['rouge1']['fmeasure']
        return row
    
    def initInfo(self):
        self.info = {}
        self.info['total_precision'] = {}
        self.info['total_recall'] = {}
        self.info['total_fmeasure'] = {}
        self.info['avg_precision'] = {}
        self.info['avg_recall'] = {}
        self.info['avg_fmeasure'] = {}
        self.info['total'] = 1
        
        for posType in self.posGroups:
            for topScorePercentage in self.topScorePrecentages:
                suffix = self.getSuffix(posType, topScorePercentage)
                self.info['total_precision'][suffix] = 0
                self.info['total_recall'][suffix] = 0
                self.info['total_fmeasure'][suffix] = 0
                
        return
    
    def summarizeInfo(self):
        for posType in self.posGroups:
            for topScorePercentage in self.topScorePrecentages:
                suffix = self.getSuffix(posType, topScorePercentage)
                self.info['avg_precision'][suffix] = self.info['total_precision'][suffix] / self.info['total']
                self.info['avg_recall'][suffix] = self.info['total_recall'][suffix] / self.info['total']
                self.info['avg_fmeasure'][suffix] = self.info['total_fmeasure'][suffix] / self.info['total']
                
                del self.info['total_precision'][suffix]
                del self.info['total_recall'][suffix]
                del self.info['total_fmeasure'][suffix]
                
                row = {}
                row['suffix'] = suffix
                row['postype'] = posType
                row['radious'] = 1.0 - topScorePercentage
                row['precision'] = self.info['avg_precision'][suffix]
                row['recall'] = self.info['avg_recall'][suffix]
                row['fmeasure'] = self.info['avg_fmeasure'][suffix]
                self.fileSummary.write(row)
        # print(self.info)
        return
    
    def getSuffix(self, posType, topScorePercentage):
        return posType + '_' + str(topScorePercentage)

    def getContributor(self, peripheralProcessor, topScorePercentage, allWords = False):
        if allWords:
            featuredWords = peripheralProcessor.getFilteredWords()
            return list(featuredWords.keys())

        peripheralProcessor.setTopScorePercentage(topScorePercentage)
        peripheralProcessor.getPoints()
        return peripheralProcessor.getContrinutors()
    
    def getPeripheralProcessor(self, text, allWords = False):
        minAllowedScore = 0 if allWords else 0.1

        peripheralProcessor = Peripheral(text)
        peripheralProcessor.setAllowedPosTypes(self.allowedTypes)
        peripheralProcessor.setPositionContributingFactor(self.positionContributingFactor)
        peripheralProcessor.setOccuranceContributingFactor(self.occuranceContributingFactor)
        peripheralProcessor.setProperNounContributingFactor(0)
        peripheralProcessor.setFilterWords(minAllowedScore)
        peripheralProcessor.loadSentences(text)
        peripheralProcessor.loadFilteredWords()
        peripheralProcessor.train()
        
        return peripheralProcessor
