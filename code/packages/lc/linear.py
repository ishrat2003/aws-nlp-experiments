from . import CWR
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

class Linear(CWR):
    
    
    def __init__(self, text, filterRate = 0.2):
        super().__init__(text, filterRate)
        self.loadSentences(text)
        return


    def train(self):
        totalWords = len(self.filteredWords)

        if totalWords == 0:
            return

        self.maxScore = 0
        for word in self.filteredWords:
            self.filteredWords[word]['score'] = self._getY(word)
            if self.filteredWords[word]['score'] > self.maxScore:
                self.maxScore = self.filteredWords[word]['score']

        return

    
    def displayPlot(self, fileName):
        #rcParams['figure.figsize']=15,10
        points = self.getPoints()
        if not points:
            print('No points to display')
            return

        plt.figure(figsize=(20, 20))  # in inches(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, edgecolors=None, *, data=None, **kwargs)[source]
        for point in points:
            plt.scatter(point['x'], point['y'], c = point['color'])
            plt.annotate(point['label'], 
                xy=(point['x'], point['y']), 
                xytext=(5, 2), 
                textcoords='offset points', 
                ha='right', 
                va='bottom')
                
        plt.xlabel('Positional importance')
        plt.ylabel('Occurrence')
        plt.savefig(fileName)
        print('After saving')
        plt.show()
        return
        
        
    def _getX(self, word):
        # Avg position when it occurred
        '''
        avg = 0
        for position in self.filteredWords[word]['positions']:
            avg += position

        avg = avg / len(self.filteredWords[word]['positions'])
        
        return avg
        '''

        # First position when occurred
        return self.filteredWords[word]['positions'][0]

        


    def _getY(self, word):
        return self.filteredWords[word]['count']

