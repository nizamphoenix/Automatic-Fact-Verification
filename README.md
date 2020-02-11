# Automatic-Fact-Verification

ML system that solves the challenge of fake news, which is unfortunately a byproduct of technological evolution. It verifies claims in the FEVER data set(citations below): each claim is labelled either SUPPORTS, REFUTES or NOTENOUGH INFO, based upon evidence in the the corpus.  

The System is evaluated on 2 metrics,  
1.the proportion of claims that have been labelled correctly(accuracy), and  
2.the correctness of a set of relevant evidence-sentences that have been identified to apparently support the label.  

Our system performs better than the baseline system for FEVER challenge which achieved a score of 33% in labelling the verdict; we address major drawbacks of the baseline system, viz. using entity matching along with inverted index for information retrieval, pair wise entailment of a claim and each of the sentences retrieved by the information retrieval system, and
employing universal sentence encoder for encoding the claim and evidence to identify at most top-scoring 5 evidences.



@inproceedings{Thorne18Fever,  
    author = {Thorne, James and Vlachos, Andreas and Christodoulopoulos, Christos and Mittal, Arpit},  
    title = {{FEVER}: a Large-scale Dataset for Fact Extraction and VERification},  
    booktitle = {NAACL-HLT},  
    year = {2018}  
}
