import setup_run_dir    # this import tricks script to run from 2 levels up
from src.utils.logging import setup_logging, WARN
from src.utils.config import Config
from src.models.parse_gsii import trainer
from src.utils.log_splitter import LogSplitter


if __name__ == '__main__':
    setup_logging(logfname='logs/train_gsii.log', level=WARN)
    args = Config.load('configs/model_parse_gsii.json')
    ls = LogSplitter('train.log')
    trainer.run_training(args, ls)
