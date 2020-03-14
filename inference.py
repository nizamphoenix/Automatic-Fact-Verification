import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity

#embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
# Create graph and finalize (finalizing optional but recommended).
g = tf.Graph()
with g.as_default():
    text_input = tf.placeholder(dtype=tf.string, shape=[None])
    sent_embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
    embedded_text = sent_embed(text_input)
    init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    g.finalize()

# Create session and initialize.
session = tf.Session(graph=g)
session.run(init_op)
# Using entaiment predicttor to get three probablities
def get_label_scores(query, sentence):
    return predictor.predict(hypothesis = query,premise = sentence)['label_probs']

def get_label(scores):
    sup, ref, noI = scores
    if max(sup, ref, noI) == sup:
        return "SUPPORTS"
    if max(sup, ref, noI) == ref:
        return "REFUTES"
    if max(sup, ref, noI) == noI:
        return "NOT ENOUGH INFO"
        
        
 pronouns_one = [' he ', 'He ', 'She ', ' she ', 'It ', ' it ']
pronouns_two = [' his ', 'His ', ' her ', 'Her ', ' its ', 'Its ']

def sort(tup): 
    return(sorted(tup, key = lambda x: float(x[0]), reverse = True))

def clean_sent(sent):
    return re.sub('\'', '', re.sub( " -RRB-",')', re.sub("-LRB- ",'(',re.sub(" $" , '', re.sub('.\n','', sent)))))

scores_list = []

def label_and_evidenve(query, k):
    result = find_possible_sentences(query, 5)
    messages = []
    ids = []
    
    for docid,pair in result:
        norm_id = unicodedata.normalize('NFD',re.sub('_',' ', re.sub('-LRB-(\w)+-RRB-','', pair[0])))
        sent = clean_sent(all_wiki_sentences[docid])
        for word in pronouns_one:
            if word in sent:
                sent = re.sub(word, ' '+ norm_id + ' ', sent)
        for word in pronouns_two:
            if word in sent:
                sent = re.sub(word, ' ' + norm_id + '\'s ', sent)    
        messages.append(re.sub('.\n','',sent))
        ids.append(identifier[docid])

    
    sent_embeddings = [*map(lambda sent:(session.run(embedded_text, feed_dict={text_input: [sent]})), messages)]
    query_embedding = session.run(embedded_text, feed_dict={text_input: [query]})

    # Sentences re-ranking using embedding
    similarities = [*map(lambda sent_embedding:(cosine_similarity(sent_embedding, query_embedding).flatten()), sent_embeddings)]
    sorted_pair = sort(list(zip([list(i)[0] for i in similarities], list(range(len(messages))))))
    
    #evidence = []
    unlabeled_sents_ids = []
    sim = []
    unlabeled_sents = []
    output = []
    for score,index in sorted_pair[:k]:
        #evidence.append(sent_embeddings[index])
        unlabeled_sents_ids.append(ids[index])
        sim.append(score)
        unlabeled_sents.append(messages[index])
        
    scores = [*map(lambda sent:(get_label_scores(query, sent)), unlabeled_sents)]

    output = [] 
    i = 0
    for sup, ref, noI in scores:
        output.append((sim[i], sup, ref, noI), (unlabeled_sents_ids[i]))
        i += 1

    return output
    
    
import json
with open('test-unlabelled.json') as f:
    test = json.load(f)
    
    
    
def aggregate_preds(predictions, top_one_sent=False):
      # Reference: https://github.com/uclmr/fever/blob/master/jack_reader.py

    """return the most popular label
    """
    vote = dict()
    preds = []
    for pred in predictions[0]:
        preds.append(["SUPPORTS", "NOT ENOUGH INFO", "REFUTES"][np.argmax(pred)])

    for pred in preds:
        if pred not in vote:
            vote[pred] = 1
        else:
            vote[pred] += 1
    
    supports = "SUPPORTS"
    refutes = "REFUTES"
    nei = "NOT ENOUGH INFO"
    
    # believe more-likely evidence if both supports and refutes appears in the pred_list
    if supports in vote and refutes in vote:
        for pred in preds:
            if pred in [supports, refutes]:
                final_verdict = pred
                break
    elif supports in vote:
        final_verdict = supports
    elif refutes in vote:
        final_verdict = refutes
    else:
        final_verdict = nei
    if top_one_sent:
        final_verdict = preds[0]

     

    if final_verdict != nei:
        for temp in predictions[1]:
            page_id, sent_num = temp
            try:
                evidence = (page_id, int(sent_num))
                evidences.append(evidence)
            except:
                    # Some values of sent_nums are string and could not be 
                    # typecasted to integer, we will store the original value.
                print("Error")
                evidences.append(temp)
                continue

    return (final_verdict, evidences)
