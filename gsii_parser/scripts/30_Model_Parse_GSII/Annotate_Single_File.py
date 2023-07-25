# Annotate the raw AMR entries with SpaCy to add the required ::tokens and ::lemmas fields
# plus a few other fields for future pre/postprocessing work that may be needed.
if __name__ == '__main__':
    import setup_run_dir    # this import tricks script to run from 2 levels up
    import os
    from   amrlib.graph_processing.annotator_v2 import annotate_file, load_annotator_model
    from   amrlib.utils.logging import silence_penman

    silence_penman()
    indir  = '../data/AMR/LDC2020T02'
    outdir = '../data/AMR/LDC2020T02'

    # Create the processed corpus directory
    os.makedirs(outdir, exist_ok=True)

    # Load the spacy model with the desired model
    load_annotator_model("cahya/bert-base-indonesian-NER")

    # run the pipeline
    for fn in (['dev.txt', 'train.txt', 'test.txt']):
        annotate_file(indir, fn, outdir, fn + '.features')
