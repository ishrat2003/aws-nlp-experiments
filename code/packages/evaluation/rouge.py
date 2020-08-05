'''
https://github.com/google-research/google-research/tree/master/rouge
'''
from rouge_score import rouge_scorer

class Rouge:

    def __init__(self, params, keys = None):
        self.params = params
        self.keys = ['rouge1', 'rouge2', 'rouge3', 'rougeL'] if not keys else keys
        self.scorer = rouge_scorer.RougeScorer(self.keys, use_stemmer=True)
        return
    
    def getScore(self, target, generated):
        score = self.scorer.score(target, generated)
        processedScores = {}
        for key in self.keys:
            processedScores[key] = {}
            processedScores[key]['precision'] = getattr(score[key], 'precision')
            processedScores[key]['recall'] = getattr(score[key], 'recall')
            processedScores[key]['fmeasure'] = getattr(score[key], 'fmeasure')
        return processedScores
