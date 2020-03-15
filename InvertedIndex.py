class InvertedIndex:
    """
    A class representating an inverted-index.
    ..........................................

    Attributes
    ----------
    vocab : dict
         a dictionary of word:id as key-value pair
    doc_term_freqs : list
        a list of doc term frequencies


    Methods
    -------
    num_terms():
        returns the vocabulary size of the corpus
    num_docs():
        returns the number of documents in the corpus
    docids(term):
        returns the document ids that contain *term* in the corpus
    freqs(term):
        returns a list of frequencies of *term* in all documents in the corpus
    f_t(term):
        returns the frequency of *term* in all documents in the corpus
    space_in_bytes():
        returns the size of inverted-index created
    """
   
    def __init__(self, vocab, doc_term_freqs):
        self.vocab = vocab
        self.doc_len = [0] * len(doc_term_freqs)
        self.doc_term_freqs = [[] for i in range(len(vocab))]
        self.doc_ids = [[] for i in range(len(vocab))]
        self.doc_freqs = [0] * len(vocab)
        self.total_num_docs = 0
        self.max_doc_len = 0
        for docid, term_freqs in enumerate(doc_term_freqs):
            doc_len = sum(term_freqs.values())
            self.max_doc_len = max(doc_len, self.max_doc_len)
            self.doc_len[docid] = doc_len
            self.total_num_docs += 1
            for term, freq in term_freqs.items():
                try:
                    term_id = vocab[term]
                    self.doc_ids[term_id].append(docid)
                    self.doc_term_freqs[term_id].append(freq)
                    self.doc_freqs[term_id] += 1
                except KeyError:
                    continue

    def num_terms(self):
        return len(self.doc_ids)

    def num_docs(self):
        return self.total_num_docs

    def docids(self, term):
        term_id = self.vocab[term]
        return self.doc_ids[term_id]

    def freqs(self, term):
        term_id = self.vocab[term]
        return self.doc_term_freqs[term_id]

    def f_t(self, term):
        term_id = self.vocab[term]
        return self.doc_freqs[term_id]

    def space_in_bytes(self):
        # this function assumes each integer is stored using 8 bytes
        space_usage = 0
        for doc_list in self.doc_ids:
            space_usage += len(doc_list) * 8
        for freq_list in self.doc_term_freqs:
            space_usage += len(freq_list) * 8
        return space_usage
