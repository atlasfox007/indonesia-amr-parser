# Annotate the raw AMR entries with SpaCy to add the required ::tokens and ::lemmas fields
# plus a few other fields for future pre/postprocessing work that may be needed.
if __name__ == '__main__':
    import setup_run_dir    # this import tricks script to run from 2 levels up
    import os
    from   src.graph_processing.annotator_v2 import annotate_file, load_annotator_model
    from   src.utils.logging import silence_penman

    silence_penman()
    indir  = '../data/AMR/amr_gold_indonesia'
    outdir = '../data/AMR/amr_gold_indonesia'

    # Create the processed corpus directory
    os.makedirs(outdir, exist_ok=True)

    # Load the model 
    load_annotator_model("cahya/bert-base-indonesian-NER")

    # run the pipeline
    for fn in (['amr_simple_test.txt', 'amr_simple_dev.txt', 'amr_simple_train.txt']):
        annotate_file(indir, fn, outdir, fn + '.features', amr_type="gold_simple")
