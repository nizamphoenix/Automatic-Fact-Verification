
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K
import os
from math import floor, ceil
from transformers import *

def get_embed(str1, str2, truncation_strategy, length):
        '''
        reuturns embeddings of str1+str2
        '''
        inputs = tokenizer.encode_plus(str1, str2,
            add_special_tokens=True,
            max_length=length,
            truncation_strategy=truncation_strategy)
        
        input_ids =  inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)
        return [input_ids, input_masks, input_segments]

def get_transformer_inputs(claim, evidence_list, tokenizer, max_sequence_length):
    input_ids_claim, input_masks_claim, input_segments_claim = get_embed(
        claim, None, 'longest_first', max_sequence_length)

    evidence= ''.join(evidence_list)
    input_ids_evid, input_masks_evid, input_segments_evid = get_embed(
        evidence, None, 'longest_first', max_sequence_length)
    
    return [input_ids_claim, input_masks_claim, input_segments_claim,
            input_ids_evid, input_masks_evid, input_segments_evid]

def get_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids_claim, input_masks_claim, input_segments_claim = [], [], []
    input_ids_evid, input_masks_evid, input_segments_evid = [], [], []
        
    for _, row in tqdm(df[columns].iterrows(),total=len(df)):
        claim, evidence_list = row.claim, row.evidence

        ids_claim, masks_claim, segments_claim, ids_evid, masks_evid, segments_evid = \
        get_transformer_inputs(claim,evidence_list tokenizer, max_sequence_length)
        
        input_ids_claim.append(ids_claim)
        input_masks_claim.append(masks_claim)
        input_segments_claim.append(segments_claim)

        input_ids_evid.append(ids_evid)
        input_masks_evid.append(masks_evid)
        input_segments_evid.append(segments_evid)
        
    return [np.asarray(input_ids_claim, dtype=np.int32), 
            np.asarray(input_masks_claim, dtype=np.int32), 
            np.asarray(input_segments_claim, dtype=np.int32),
            np.asarray(input_ids_evid, dtype=np.int32), 
            np.asarray(input_masks_evid, dtype=np.int32), 
            np.asarray(input_segments_evid, dtype=np.int32)]

def get_output_arrays(df, columns):
    return np.asarray(df[columns])
