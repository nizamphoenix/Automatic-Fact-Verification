from tqdm import tqdm_notebook
from tqdm.notebook import tqdm
import os
def make_data(path):
    filenames = sorted(os.listdir(path))
    all_wiki_sentences = []
    identifier = []
    for filename in tqdm(filenames):#looping 109 files
        with open('./data/wiki/wiki-pages-text/'+filename) as wikifile:#opening 1 file at a time
            for sent in wikifile.readlines():
                sent = sent.rstrip()
                sent = sent.replace('-LRB-',"(")
                sent = sent.replace('-RRB-',")")
                page_id,sent_num,sent = sent.split(" ",2)
                page_id = page_id.replace('_',' ')
                all_wiki_sentences.append(sent)
                identifier.append((page_id,sent_num))
    print(len(all_wiki_sentences),len(all_wiki_sentences)==len(identifier))
    return identifier,all_wiki_sentences





import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import unicodedata
import string
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english') + list(string.punctuation)+list(('``','--'))+list(("-lrb-", "-rrb-"))
nltk.download('punkt')

def lemmatize_my_token(word):
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(word,'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word,'n')
    return lemma

def normalize_token(token):
    nfd_token = unicodedata.normalize('NFD',token)
    norm_token = lemmatize_my_token(nfd_token)
    return norm_token

def get_normalized_sentence(sent):
    '''
      1.all tokens lowercased
      2.NFD normalization applied
      3.lemmatized
      4.stop words and punctuations excluded
    '''
    norm_sent = []#contains normalized tokens of a single sentence(treated as a document)      
    tokens = {*map(lambda token:normalize_token(token),nltk.word_tokenize(sent.lower()))}
    norm_sent = list(tokens.difference(set(stop).intersection(tokens)))
    return norm_sent


    

