# Automatic-Fact-Verification
A Machine learning system that solves the challenge of fake news detection by utilizing Natural Language Processing techniques: Information Retrieval, Machine Reading Comprehension, and textual entailment.

The Information retrieval system processes the corpus and builds an index for lookup; Machine comprehension system imbibes understanding language with the help of embeddings(ELMo) and provides logically coherent sentences; finally, textual entailment checks if the claim can be inferred from the evidence.  

The famous **FEVER** data set(citations below) is used as corpus to build the verification system.  
A claim posited is either supported, refuted, or annulled depending upon evidence found in the corpus, it is annulled if no evidence is found either supporting or refuting the claim being made.  

The System is evaluated on 2 metrics,  
1. The proportion of claims that have been labeled correctly(accuracy), and  
2. The correctness of a set of relevant sentences(evidence) that have been identified to, apparently, support the label.  

Our system achieves a score of 45.14% and performs better than the baseline system for FEVER challenge which achieved a score of 33% in labeling the verdict; we address major drawbacks of the baseline system, using entity matching along with an inverted index for information retrieval, pairwise entailment of a claim and each of the sentences retrieved by the information retrieval system, and employing universal sentence encoder for encoding the claim and relevant evidence to identify at most top-scoring 5 statements(evidence).  

Currenty, built with simple inverted index, Google Universal sentence encoder & ELMo embedding.  
Future work: use better sentence encoders & BERT embeddings, also sophisticated tools for building inverted index.
### Citation
```
@inproceedings{Thorne18Fever,  
    author = {Thorne, James and Vlachos, Andreas and Christodoulopoulos, Christos and Mittal, Arpit},  
    title = {{FEVER}: a Large-scale Dataset for Fact Extraction and VERification},  
    booktitle = {NAACL-HLT},  
    year = {2018}  
}
```
