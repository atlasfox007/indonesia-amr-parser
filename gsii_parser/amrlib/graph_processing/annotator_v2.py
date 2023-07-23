import os
import json
import logging
import penman
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForTokenClassification
from   .amr_loading import load_amr_entries

logger = logging.getLogger(__name__)

# Default set of tags to keep when annotating the AMR. Throw all others away
# To keep all, redefine this to None
keep_tags = set(['id', 'snt'])

# Annotate a file with multiple AMR entries and save it to the specified location
def annotate_file(indir, infn, outdir, outfn):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForTokenClassification.from_pretrained('bert-base-uncased')

    inpath = os.path.join(indir, infn)
    entries = load_amr_entries(inpath)

    graphs = []
    for entry in tqdm(entries, desc="Processing Entries"):
        pen = _process_entry(entry, tokenizer, model)
        graphs.append(pen)

    infn = infn[:-3] if infn.endswith('.gz') else infn  # strip .gz if needed
    outpath = os.path.join(outdir, outfn)
    print('Saving file to ', outpath)
    penman.dump(graphs, outpath, indent=6)


# Annotate a single AMR string and return a penman graph
def annotate_graph(entry, tokenizer, model):
    return _process_entry(entry, tokenizer, model)


# Process a single AMR entry using the tokenizer and model
def _process_entry(entry, tokenizer, model):
    pen = penman.decode(entry)  # standard de-inverting penman loading process
    tokens = tokenizer.tokenize(pen.metadata['snt'])
    inputs = tokenizer(pen.metadata['snt'], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    ner_tags = torch.argmax(outputs.logits, dim=2)[0].tolist()  # Get predicted NER tags for the first (and only) sentence
    pen.metadata['tokens'] = json.dumps(tokens)
    pen.metadata['ner_tags'] = json.dumps(ner_tags)
    return pen
