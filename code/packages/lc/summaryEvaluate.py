from .basicEvaluate import BasicEvaluate
from .peripheral import Peripheral

class SummaryEvaluate(BasicEvaluate):

    def getFileName(self, prefix = ''):
        return self.params.dataset_name + '_' + prefix + '_summary_pos' + str(self.positionContributingFactor) + '_occ' + str(self.occuranceContributingFactor) + '.csv';

    def process(self, store = True):
        self.initInfo()
        data = self.dataset.getTrainingSet()
        for (batch, (source, target)) in enumerate(data):
            sourceText = source.numpy().decode('utf-8')
            targetText = target.numpy().decode('utf-8')
            row = self.processItem(batch, sourceText, targetText)
            self.file.write(row)
            self.info['total'] += 1
            
        self.summarizeInfo()
        return

    def processItem(self, batch, sourceText, targetText):
        seperator = ' '
        if self.params.display_details:
            print('Batch:::::::::::::::::: ', batch)
            print('Content::: ', sourceText)
            print('Summary::: ', targetText)
        
        row = {}
        # row['main_text'] = sourceText
        # row['summary_text'] = targetText
        
        for posType in self.posGroups:
            self.setAllowedTypes(self.posGroups[posType])
            peripheralProcessor = self.getPeripheralProcessor(sourceText)
            
            for topScorePercentage in self.topScorePrecentages:
            
                generatedContributor = self.getContributor(peripheralProcessor, topScorePercentage)
                expectedContributor = self.getContributor(peripheralProcessor, 0, True)
                expectedContributor = self.dataset.getProcessedText(expectedContributor)
            
                generatedContributor = seperator.join(generatedContributor)
                expectedContributor = seperator.join(expectedContributor)
                
                values = self.evaluate(generatedContributor, expectedContributor, posType, topScorePercentage)
                row.update(values)
                   
        return row
