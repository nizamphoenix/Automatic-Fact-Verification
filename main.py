import utils
import preproc
import retrieve
import train
import inference
import multiprocessing 

#for the first-half of sentences
processed_sents1 = []
ncores = 16 #was run on GCP 
pool = multiprocessing.Pool(ncores)
processed_sents1 = pool.map(get_normalized_sentence,tqdm(all_wiki_sentences[:12624198]))
pool.close()
#for the second-half of sentences
processed_sents2 = []
ncores = 16 
pool = multiprocessing.Pool(ncores)
processed_sents2 = pool.map(get_normalized_sentence,tqdm(all_wiki_sentences[12624198:]))
pool.close()
processed_sents = []
processed_sents.extend(processed_sents1)
processed_sents.extend(processed_sents2)
len(processed_sents) == len(processed_sents1)+len(processed_sents2)

doc_term_freq,vocab=create_doc_term_freq(filepath)

# doc_term_freqs stores the counters (mapping terms to term frequencies) of all documents
doc_term_freqs = []
st = time.time()
for sent in processed_sents:
    tfs = Counter()
    for token in sent:
        tfs[token] += 1
    doc_term_freqs.append(tfs)
et = time.time()
print("Time lapsed: ",(et - st)/60.0," mins") 
print(len(doc_term_freqs))
print(doc_term_freqs[0])
print(doc_term_freqs[-109])

invindex = InvertedIndex(vocab, doc_term_freqs)
save_object(invindex, 'invindex_final.pkl')
invindex=load_object('invindex_final.pkl')               
#compute BM25
sample= invindex.num_docs()
train_data = get_training_data_for_sample(sample)
evidences = get_evidence(train_data)
model, history = fit_model(train_data,evidences)
scores=get_label_scores(model)
with open('testOutput.json', 'w') as fp:
    json.dump(get_label(scores), fp)
