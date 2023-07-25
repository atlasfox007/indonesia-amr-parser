import os
import json
import logging
import penman
import torch
import multiprocessing
from tqdm import tqdm
from transformers import BertTokenizer, BertForTokenClassification, pipeline
from nlp_id.tokenizer import Tokenizer
from nlp_id.postag import PosTag
from nlp_id.lemmatizer import Lemmatizer 
from typing import List
from   .amr_loading import load_amr_entries

#  Loggger related
logger = logging.getLogger(__name__)

# Loading All annotator model
tokenizer_model : Tokenizer = None
ner_pipeline : pipeline  = None
postag_model : PosTag = None
lemmatizer_model : Lemmatizer = None

def load_annotator_model(ner_model_name=None):
    global ner_pipeline, tokenizer_model, postag_model, lemmatizer_model

    ner_pipeline = pipeline("token-classification", model=ner_model_name)
    tokenizer_model = Tokenizer()
    postag_model = PosTag()
    lemmatizer_model = Lemmatizer()

# Default set of tags to keep when annotating the AMR. Throw all others away
# To keep all, redefine this to None
keep_tags = set(['id', 'snt'])

# Start Method variable
start_method = None

# Annotate a file with multiple AMR entries and save it to the specified location
def annotate_file(indir, infn, outdir, outfn):
    inpath = os.path.join(indir, infn)
    entries = load_amr_entries(inpath)


    graphs = []
    global start_method
    if start_method is not None:
        multiprocessing.set_start_method(start_method)  # can not be used more than once in the program.
        start_method = None
    # Unix platforms
    if multiprocessing.get_start_method() == 'fork':
        with multiprocessing.Pool() as pool:
            for pen in tqdm(pool.imap(_process_entry, entries), total=len(entries)):
                graphs.append(pen)
    # Windows and Mac
    else:
        for pen in tqdm(map(_process_entry, entries), total=len(entries)):
            graphs.append(pen)

    infn = infn[:-3] if infn.endswith('.gz') else infn  # strip .gz if needed
    outpath = os.path.join(outdir, outfn)
    print('Saving file to ', outpath)
    penman.dump(graphs, outpath, indent=6)

# Process a single AMR entry using the tokenizer and model
def _process_entry(entry : str):
    pen = penman.decode(entry)  # standard de-inverting penman loading process
    return _process_penman(pen)

def _process_penman(pen : penman.Graph):
    # Filter out old tags and add the tags from SpaCy parse
    global keep_tags
    if keep_tags is not None:
        pen.metadata = {k:v for k,v in pen.metadata.items() if k in keep_tags}  # filter extra tags
    
    # Tokenization
    tokens = process_token(pen.metadata['snt'])
    pen.metadata['tokens']   = json.dumps(tokens)
    # NER
    pen.metadata['ner_tags'] = json.dumps(process_ner(tokens, pen.metadata['snt']))
    # POSTAG
    pen.metadata['pos_tags'] = json.dumps(process_postag(tokens))
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

def process_postag(inp : List[str]) -> List[str] :
    global postag_model
    postag = postag_model.get_pos_tag(" ".join(inp))

    postag_result = [x[1] for x in postag]
    return postag_result


def process_lemma(inp : List[str]) -> List[str] :
    global lemmatizer_model
    lemma_result = lemmatizer_model.lemmatize(" ".join(inp))
    return lemma_result.split()