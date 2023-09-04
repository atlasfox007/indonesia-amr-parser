import os
import json
import logging
import penman
import torch
import multiprocessing
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from nlp_id.tokenizer import Tokenizer
from nlp_id.postag import PosTag
from nlp_id.lemmatizer import Lemmatizer 
from typing import List
from   .amr_loading import load_amr_entries, load_indo_news_amr_entries, load_simple_amr_indonesia

#  Loggger related
logger = logging.getLogger(__name__)

# Loading All annotator model
tokenizer_model : Tokenizer = None
ner_pipeline : pipeline  = None
postag_model : PosTag = None
lemmatizer_model : Lemmatizer = None
ner_model_name : str = None

def load_annotator_model(model_name=None):
    global ner_pipeline, tokenizer_model, postag_model, lemmatizer_model, ner_model_name
    
    ner_model_name = model_name

    if ner_pipeline is not None:
        return
    else:
        # import pickle
        # with open("../model_dump/ner_model_pipeline.pkl", 'rb') as f:
        #     ner_pipeline = pickle.load(f)
        tokenizer = AutoTokenizer.from_pretrained("cahya/bert-base-indonesian-NER")
        model = AutoModelForTokenClassification.from_pretrained("cahya/bert-base-indonesian-NER")
        ner_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer,device=0)

    if tokenizer_model is not None:
        return
    else:
        import pickle
        with open("../model_dump/tokenizer_model.pkl", 'rb') as f:
            tokenizer_model = pickle.load(f)
        # tokenizer_model = Tokenizer()

    if postag_model is not None:
        return
    else:
        import pickle
        with open("../model_dump/postag_model.pkl", 'rb') as f:
            postag_model = pickle.load(f)
        # postag_model = PosTag()
    
    if lemmatizer_model is not None:
        return
    else:
        import pickle
        with open("../model_dump/lemma_model.pkl", 'rb') as f:
            lemmatizer_model = pickle.load(f)
        # lemmatizer_model = Lemmatizer()

# Default set of tags to keep when annotating the AMR. Throw all others away
# To keep all, redefine this to None
keep_tags = set(['id', 'snt'])

# Start Method variable (can be spawn, fork)
start_method = "spawn"

# Annotate a file with multiple AMR entries and save it to the specified location
def annotate_file(indir, infn, outdir, outfn, amr_type=None):
    load_annotator_model()
    inpath = os.path.join(indir, infn)
    if(amr_type == "gold"):
        entries = load_amr_entries(inpath)
    elif(amr_type == "gold_simple"):
        entries = load_simple_amr_indonesia(inpath)
    else:
        entries = load_indo_news_amr_entries(inpath)


    graphs = []
    global start_method
    if start_method is not None:
        multiprocessing.set_start_method(start_method)  # can not be used more than once in the program.
        start_method = None
    # Unix platforms
    if multiprocessing.get_start_method() == 'fork':
        with multiprocessing.Pool(processes=4) as pool:
            for pen in tqdm(pool.imap(_process_entry, entries), total=len(entries)):
                graphs.append(pen)
    # Windows and Mac
    else:
        for pen in tqdm(map(_process_entry, entries), total=len(entries)):
            if(pen == None):
                continue
            graphs.append(pen)

    infn = infn[:-3] if infn.endswith('.gz') else infn  # strip .gz if needed
    outpath = os.path.join(outdir, outfn)
    print('Saving file to ', outpath)
    penman.dump(graphs, outpath, indent=6)

# Process a single AMR entry using the tokenizer and model
def _process_entry(entry : str):
    try:
        pen = penman.decode(entry)  # standard de-inverting penman loading process
    except:
        return
    return _process_penman(pen)

def _process_penman(pen : penman.Graph):
    global keep_tags
    if keep_tags is not None:
        pen.metadata = {k:v for k,v in pen.metadata.items() if k in keep_tags} 
    
    # Tokenization, with checking if amr representation is a
    tokens = process_token(pen.metadata['snt'])
    pen.metadata['tokens']   = json.dumps(tokens)
    # NER
    pen.metadata['ner_tags'] = json.dumps(process_ner(tokens, pen.metadata['snt']))
    # POSTAG
    pen.metadata['pos_tags'] = json.dumps(process_postag(pen.metadata['snt']))
    # LEMMA
    pen.metadata['lemmas'] = json.dumps(process_lemma(tokens))
    return pen

def process_token(inp : str) -> List[str]:
    return tokenizer_model.tokenize(inp)

def process_ner(inp_token : List[str], input_words : str) -> List[str]:
    global ner_pipeline


    ner_pipeline_result = ner_pipeline(input_words)

    ner_dict = {}

    for i in range(len(ner_pipeline_result)):
        ner_dict[ner_pipeline_result[i]['word']] = ner_pipeline_result[i]['entity']
    
    ner_result = []

    for word in inp_token:
        word_check = word.lower()
        if(word_check in ner_dict):
            ner_result.append(ner_dict[word_check])
        else:
            ner_result.append('O') # Not present in ner_pipeline_result

    return ner_result

def process_postag(input_word : str) -> List[str] :
    global postag_model
    postag = postag_model.get_pos_tag(input_word)

    postag_result = [x[1] for x in postag]
    return postag_result


def process_lemma(inp : List[str]) -> List[str] :
    global lemmatizer_model
    lemma_result = []

    for word in inp:
        lm = lemmatizer_model.lemmatize(word)
        if(len(lm) < 1):
            lemma_result.append(word)
        else:
            lemma_result.append(lm)
    return lemma_result