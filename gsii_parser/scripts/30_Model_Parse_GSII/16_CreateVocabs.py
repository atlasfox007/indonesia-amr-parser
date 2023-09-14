import setup_run_dir    # this import tricks script to run from 2 levels up
import os
from   src.utils.logging import setup_logging, WARN
from   src.models.parse_gsii.create_vocabs import create_vocabs


# Create vocabs from the training data
if __name__ == '__main__':
    setup_logging(logfname='logs/create_vocabs.log', level=WARN)
    train_data = '../data/AMR/amr_gold_silver_combined/amr_gold_silver_combined.txt.features'
    vocab_dir  = '../pretrained_model_indonesia_gold_silver_combined/vocabs'

    os.makedirs(vocab_dir, exist_ok=True)

    create_vocabs(train_data, vocab_dir)
