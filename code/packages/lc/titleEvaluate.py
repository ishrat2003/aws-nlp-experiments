from .basicEvaluate import BasicEvaluate
from .peripheral import Peripheral

class TitleEvaluate(BasicEvaluate):
    
    def getFileName(self, prefix = ''):
        return self.params.dataset_name + '_' + prefix + '_title_pos' + str(self.positionContributingFactor) + '_occ' + str(self.occuranceContributingFactor) + '.csv';

    def setAllowedTypes(self, allowedTypes):
        self.allowedTypes = allowedTypes
        return
    
    def processItem(self, batch, item):
        label = self.dataset.getLabel(item)
        sourceText = self.dataset.getText(item)

        seperator = ' '
        if self.params.display_details:
            print('Batch:::::::::::::::::: ', batch)
            print('Content::: ', sourceText)
        
        row = {}
        
        for posType in self.posGroups:
            self.setAllowedTypes(self.posGroups[posType])
            
            expectedContributor = self.dataset.getTitle(item)
            expectedContributor = self.dataset.getProcessedText(expectedContributor)
            
            if self.params.display_details:
                print('Title topics::: ', expectedContributor)

            peripheralProcessor = self.getPeripheralProcessor(sourceText)
                
            for topScorePercentage in self.topScorePrecentages:
                cwrGeneratedContributor = self.getContributor(peripheralProcessor, topScorePercentage)
                cwrGeneratedContributor = seperator.join(cwrGeneratedContributor)
                values = self.evaluate(cwrGeneratedContributor, expectedContributor, posType, topScorePercentage)
                row.update(values)
                
        return row
    