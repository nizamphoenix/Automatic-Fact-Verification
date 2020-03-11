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

def extract_term_freqs(preprocessed_sent):
    tfs = Counter()    
    for token in preprocessed_sent:
        tfs[token] += 1
    return tfs

def bm_25(query, index, k):      
    N = invindex.num_docs()
    Lavg = sum(invindex.doc_len)/N
    scores_bm25 = Counter()
    #scores_tfidf = Counter()
    query_terms = preprocessed_sentence(query)
    query_terms_freqs = extract_term_freqs(query_terms)
    k1 = 1.2
    k3 = 1.5
    b = 0.75
    
    def fill_scores_one(term):
        try:
            f_term = index.f_t(str(term))         #number of docs containing the term 
            docids = index.docids(str(term))    #list of Ids of the documents containing the term 't'
            fqt = query_terms_freqs[str(term)]   # number of term in a query
            
            
            def fill_scores_two(d, docid): 
           
            #for d,docid in enumerate(docids):   
            
                fdt =  index.freqs(term)[d]  # number of a term in a sentence
                Ld = index.doc_len[docid]    # length of a sentence
                idf = log((N - f_term + 0.5)/(f_term + 0.5))
                tf_doc = ((k1 + 1) * fdt)/(k1 * ((1-b) + b * Ld/Lavg) + fdt)
                query_tf = (k3 + 1) * fqt / (k3 + fqt)
                wt = idf * tf_doc * query_tf
                scores_bm25[docid] += wt
            
            temp = [*map(fill_scores_two, list(range(len(docids))), docids)]
            temp = []
             
                #temp = 0 #an auxiliary variable to aid in computing the score.
                #temp = temp + log(1 + fdt) * log(N/float(f_term))
                #scores_tfidf[docid] += 1.0/sqrt(index.doc_len[docid]) * temp
        except KeyError:
            pass
        except IndexError:
            print("Index error occured. Try again later.")

    
    dummy = [*map(fill_scores_one, query_terms)]
    dummy = []
    
    #final_docids = set()
    #for x in scores_bm25.most_common(k):
        #final_docids.add(x[0])
    #for x in scores_tfidf.most_common(k):
        #final_docids.add(x[0])
    return [x[0] for x in scores_bm25.most_common(k)]
