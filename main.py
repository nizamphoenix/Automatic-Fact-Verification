import preproc
import misc
doc_term_freq,vocab=create_doc_term_freq(filepath)
invindex = InvertedIndex(vocab, doc_term_freqs)
save_object(invindex, 'invindex_final.pkl')
#invindex=load_object('invindex_final.pkl')               
