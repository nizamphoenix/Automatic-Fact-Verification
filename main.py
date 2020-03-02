import utils
import preproc
import retrieve
import train
import inference

doc_term_freq,vocab=create_doc_term_freq(filepath)
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
