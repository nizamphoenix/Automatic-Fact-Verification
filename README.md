# Automatic-Fact-Verification

ML system that solve the challenge of fake news,which is unfortunately a byproduct of technological evolution by verifying claims in the FEVER data set(citations below). Each claim islabelled as SUPPORTS, REFUTES or NOTENOUGH INFO, depending on whether any statements in the corpus substantiate it, accordingly.  

The System is evaluated on 2 metrics,  
1.the proportion of claims that have been labelled correct(accuracy), and  
2.the correctness of a complete set of relevant evidence sentences that have been identified to support the label.  

Our system performs better than the baseline system \cite{Thorne18Fever} for FEVER challenge which achieves a score of 33% in labelling the verdict as we address major drawbacks of the baseline system, viz. using entity matching along with inverted index for information retrieval, pair wise entailment of a claim and each of the sentences retrieved by the IR system, and
employing universal sentence encoder for encoding the claim and evidence to identify at most top-scoring 5 evidences.



@inproceedings{Thorne18Fever,  
    author = {Thorne, James and Vlachos, Andreas and Christodoulopoulos, Christos and Mittal, Arpit},  
    title = {{FEVER}: a Large-scale Dataset for Fact Extraction and VERification},  
    booktitle = {NAACL-HLT},  
    year = {2018}  
}
