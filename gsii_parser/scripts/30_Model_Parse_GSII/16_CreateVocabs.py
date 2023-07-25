import setup_run_dir    # this import tricks script to run from 2 levels up
import os
from   amrlib.utils.logging import setup_logging, WARN
from   amrlib.models.parse_gsii.create_vocabs import create_vocabs


# Create vocabs from the training data
if __name__ == '__main__':
    setup_logging(logfname='logs/create_vocabs.log', level=WARN)
    train_data = '../data/AMR/LDC2020T02/train.txt.features.nowiki'
    vocab_dir  = '../pretrained_model/vocabs'

    os.makedirs(vocab_dir, exist_ok=True)

    create_vocabs(train_data, vocab_dir)
