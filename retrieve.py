def extract_term_freqs(preprocessed_sent):
    tfs = Counter()    
    for token in preprocessed_sent:
        tfs[token] += 1
    return tfs

#compute BM25


# query: a claim index: the inverted index   k: the maximun number of sentences returned
def bm_25(query, index, k):      
    N = index.num_docs()
    Lavg = sum(index.doc_len)/N
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
    
 # Create a dictionary of processed identifiers
proccessed_ids = defaultdict(list)
for doc_id, pair in enumerate(identifier):
    proccessed_id = unicodedata.normalize('NFD',re.sub('_',' ', re.sub('-(\w)+-','', pair[0])).lower())
    proccessed_id[proccessed_id].append((pair[1], doc_id))
    
    
    
 # Create a dictionary of processed identifiers
proccessed_ids = defaultdict(list)
for doc_id, pair in enumerate(identifier):
    proccessed_id = unicodedata.normalize('NFD',re.sub('_',' ', re.sub('-(\w)+-','', pair[0])).lower())
    proccessed_id[proccessed_id].append((pair[1], doc_id))
    
    
# Types of the entities in the claim
type_words = ['film', 'movie', 'song', 'band', 'television series', 'novel', \
              'comics', 'magazine', 'play', 'soundtrack']

def find_ents_ids(query):
    
    query = unicodedata.normalize('NFD',query)
    evidence_ids = []
    
    # Search possible type words
    query_type = None
    for type_word in type_words:
        if type_word in query.lower():
            if type_word == 'movie':
                query_type = 'film'
                break
            if type_word == 'television series':
                query_type = 'tv series'
                break
            query_type = type_word
            break   
    
    # Find capitaltized tokens
    temp = []
    num_cap = 0
    cap_token = []
    tokens = word_tokenize(query)
    length_query = len(tokens)
    i = 0
    for token in tokens:
        i += 1
        if token[0].isupper():
            num_cap += 1
            cap_token.append(token)
            if i == length_query:
                temp.append(' '.join(cap_token))
        else:
            if num_cap == 0:
                continue
            else:
                if cap_token[-1] in ['of', 'a', 'the']:
                    cap_token.append(token)
                elif token in ['of', 'a', 'the'] and token != cap_token[-1]:
                    cap_token.append(token)
                elif token.isnumeric():
                    cap_token.append(token)  
                else:
                    if num_cap > 0:
                        temp.append(' '.join(cap_token))
                        num_cap = 0
                        cap_token = []

    cap_tokens = [re.sub('the ', '', token.lower()) for token in temp]
    #print(cap_tokens)
    
    def find_matching(item):
        return list(filter(lambda x:item in x, list(proccessed_ids.keys())))

    # Find identifiers that contain the capitaltized tokens
    ids_list = [*map(find_matching, cap_tokens)]
    i = 0
    for ids in ids_list:
        if len(ids) <= 3:
            evidence_ids.extend(ids)
        else:
            if query_type != None:
                item = cap_tokens[i]+" "+query_type
                evidence_ids.extend(list(filter(lambda x: item in x, ids)))  
            else:
                if len(evidence_ids) == 0:
                    item = cap_tokens[i]
                    evidence_ids.extend(list(filter(lambda x : x == item, ids)))
        i += 1
    return evidence_ids

#  Find all document ids for all identifiers
def find_ents_sentences(query):
    ids = find_ents_ids(query)
    evidence_docids = []
    for id in ids:
        for _,doc_id in proccessed_ids[id]:
            evidence_docids.append(doc_id)
    return evidence_docids




# Combine the result of the two methods
def find_possible_sentences(query, k):    
    results_bm25 = bm_25(query, invindex, k)
    results_ents = find_ents_sentences(query)
    sents = set()
    for res in results_bm25:
        sents.add((res, identifier[res]))
    for res in results_ents:
        sents.add((res, identifier[res]))
    return sents

# Combine the result of the two methods
def find_possible_sentences(query, k):    
    results_bm25 = bm_25(query, invindex, k)
    results_ents = find_ents_sentences(query)
    sents = set()
    for res in results_bm25:
        sents.add((res, identifier[res]))
    for res in results_ents:
        sents.add((res, identifier[res]))
    return sents




def get_evidence_docid(evidence):
    evidence_id = unicodedata.normalize('NFD',re.sub('_',' ', re.sub('-(\w)+-','', evidence[0])).lower())
    evidence_sent_num = str(evidence[1])
    for sent_num, docid in proccessed_ids[evidence_id]:
        if evidence_sent_num == sent_num:
            return docid
            break

# Generating 
def get_evidence_output(docid, query):
    
    norm_id = unicodedata.normalize('NFD',re.sub('_',' ', re.sub('-LRB-(\w)+-RRB-','', identifier[docid][0])))
    sent = clean_sent(all_wiki_sentences[docid])
    for word in pronouns_one:
        if word in sent:
            sent = re.sub(word, ' '+ norm_id + ' ', sent)
    for word in pronouns_two:
        if word in sent:
            sent = re.sub(word, ' ' + norm_id + '\'s ', sent)    
    sent = re.sub('.\n','',sent)
    #print(sent)
    
    sent_embedding = session.run(embedded_text, feed_dict={text_input: [sent]})
    query_embedding = session.run(embedded_text, feed_dict={text_input: [query]})
    sim = list(cosine_similarity(sent_embedding, query_embedding).flatten())
    score = get_label_scores(query, sent)
    #print(score)
    return (sim[0], score[0], score[1], score[2])

def get_training_data_for_sample(sample):
    LIMIT = 4
    if sample['label'] == 'NOT ENOUGH INFO':
        output = label_and_evidenve(sample['claim'] , LIMIT)
    else:
        docids = [get_evidence_docid(evidence) for evidence in sample['evidence'][:LIMIT]]
        if len(docids) == 4:
            output = [*map(lambda docid:(get_evidence_output(docid, sample['claim'])), docids)]
        else:
            output_a = [*map(lambda docid:(get_evidence_output(docid, sample['claim'])), docids)]
            if len(docids) == 1:
                output_b = [*map(lambda k:(get_evidence_output(docids[0] + k, sample['claim'])), list(range(1,4)))]
                output = output_a + output_b 
            if len(docids) == 2:
                output_b = [*map(lambda docid:(get_evidence_output(docid + 1, sample['claim'])), docids)]
                output = output_a + output_b 
            if len(docids) == 3:
                output_b = [get_evidence_output(docids[0] + 1, sample['claim'])]
                output = output_a + output_b
                
    return output  

