import time
import os

wiki_dict={}#global dictionary
filenames=sorted(os.listdir('./data/wiki/wiki-pages-text/'))
for filename in tqdm(filenames):#looping 109 files
    with open('./data/wiki/wiki-pages-text/'+filename) as wikifile:#opening 1 file at a time
        for sent in wikifile.readlines():
            sent = sent.rstrip()
            sent = sent.replace('-LRB-',"(")
            sent = sent.replace('-RRB-',")")
            page_id,sent_num,sent = sent.split(" ",2)
            temp = []
            try:#first check if the key already exists
                temp = wiki_dict[page_id]
            except KeyError:#if the key doesn't exist and it is required to create a new one 
                pass
            finally:
                temp.append((sent_num,sent))
                wiki_dict[page_id] = temp                
                
                
def create_doc_term_freq(filepath):
    '''
    Function to create document term frequency from a corpus
    '''
    filenames = []
    for file in sorted(os.listdir(filepath)):
        filenames.append(file)
    
#     for i in range(1,10):
#         filenames.append('wiki-pages-text/wiki-00'+ str(i) + '.txt')
#     for i in range(10,100):
#         filenames.append('wiki-pages-text/wiki-0'+ str(i)+'.txt')
#     for i in range(100,110):
#         filenames.append('wiki-pages-text/wiki-'+ str(i)+'.txt')
    file_as_list = []#contains a file read from wiki-pages-text
    #wiki_sentences = []
    all_wiki_sentences = []
    identifier = []
    st = time.time()
    files_processed = 0
    #term_id = -1 # global variable for assigning unique id to word types(vocabulary)
    for filename in filenames:#looping 109 files
        with open(filename) as wikifile:#opening 1 file at a time
            file_as_list.append(wikifile.readlines())#reading 1 file into a list
        for file in file_as_list:
            for sent in file:
                page_id,sent_num,sent = sent.split(" ",2)
                #wiki_sentences.append(sent)
                all_wiki_sentences.append(sent)
                identifier.append((page_id,sent_num))
        #write_to_file(processed_sents,uid)
        #write_to_file(wiki_sentences)
        file_as_list = []
        #wiki_sentences = []
        files_processed += 1
    et = time.time()
    print("files_processed: ", files_processed)
    print("Time for processing wiki-corpus:",(et-st)/(3600.0),"hrs.")    
    #--------------------------------------------------------------------------------------------------
    st = time.time()  
    processed_sents = []
    # vocab contains (term, term id) pairs
    #vocab = {}
    # total_tokens stores the total number of tokens
    total_tokens = 0
    # total_stems stores the total number of terms in vocab
    #total_lemmas = 0

    for sent in all_wiki_sentences:
        norm_sent = preprocessed_sentence(sent)
        for token in norm_sent:
            total_tokens += 1
        processed_sents.append(norm_sent)  
    et = time.time()   
    print("Time for processing wiki-corpus:",(et-st)/(3600.0),"hrs.")
    print("Number of documents = {}".format(len(processed_sents)))
    #print("Number of unique terms = {}".format(len(vocab)))
    print("Number of tokens = {}".format(total_tokens))
    #------------------------------------------------------------------------------------------------
    # doc_term_freqs stores the counters (mapping terms to term frequencies) of all documents
    doc_term_freqs = []
    st = time.time()
    for tokens in processed_sents:
        tfs = Counter()
        for token in tokens:
            tfs[token] += 1
        doc_term_freqs.append(tfs)
    et = time.time()
    print("Time used: ",(et - st)/60.0," mins") 
    print(len(doc_term_freqs))
    print(doc_term_freqs[0])
    print(doc_term_freqs[-109])
    #-------------------------------------------------------------------------------------------------
    return doc_term_freqs,vocab
    
def lemmatize_my_token(word):
    lemmatizer=WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(word,'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word,'n')
    return lemma
    
def preprocessed_sentence(sent):
    norm_sent = []#contains normalized tokens of a single sentence(treated as a document)
    for token in nltk.word_tokenize(sent.lower()):#1.all tokens lowercased
        nfd_token = unicodedata.normalize('NFD',token)#2.NFD normalization applied
        if nfd_token not in stop:#3.stop words and punctuations excluded
            #norm_token = lemmatize_my_token(nfd_token)#4.lemmatized
            #norm_sent.append(norm_token)
            norm_sent.append(nfd_token)
    return norm_sent
