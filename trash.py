import nltk
from nltk.corpus import PlaintextCorpusReader
import sys

batchSize = 32
emb_size = 50 
hid_size = 100
corpus_root = '.\\'
myCorpus = PlaintextCorpusReader(corpus_root, 'corpusPoems.*\.txt')
startToken = '<s>'
endToken = '</s>'
unkToken = '<unk>'
padToken = '<pad>'

device = torch.device("cpu")

class progressBar:
    def __init__(self ,barWidth = 50):
        self.barWidth = barWidth
        self.period = None
    def start(self, count):
        self.item=0
        self.period = int(count / self.barWidth)
        sys.stdout.write("["+(" " * self.barWidth)+"]")
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.barWidth+1))
    def tick(self):
        if self.item>0 and self.item % self.period == 0:
            sys.stdout.write("-")
            sys.stdout.flush()
        self.item += 1
    def stop(self):
        sys.stdout.write("]\n")


def extractDictionary(corpus, limit=20000):
    pb = progressBar()
    pb.start(len(corpus))
    dictionary = {}
    for doc in corpus:
        pb.tick()
        for w in doc:
            if w not in dictionary: dictionary[w] = 0
        dictionary[w] += 1
    L = sorted([(w,dictionary[w]) for w in dictionary], key = lambda x: x[1] , reverse=True)
    if limit > len(L): limit = len(L)
    words = [ w for w,_ in L[:limit] ] + [unkToken] + [padToken]
    word2ind = { w:i for i,w in enumerate(words)}
    pb.stop()
    return words, word2ind

corpus = [ [startToken] + [w.lower() for w in sent] + [endToken] for sent in myCorpus.sents()]
words, word2ind = extractDictionary(corpus)

blm = LSTMLanguageModelPack(emb_size, hid_size, word2ind, unkToken, padToken, endToken,2, 0.5).to(device)
optimizer = torch.optim.Adam(blm.parameters(), lr=0.01)