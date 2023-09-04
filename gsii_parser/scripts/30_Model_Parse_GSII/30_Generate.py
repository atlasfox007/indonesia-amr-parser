if __name__ == '__main__':
    import os
    import setup_run_dir    # this import tricks script to run from 2 levels up
    from   src.utils.logging import setup_logging, WARN
    from   src.models.parse_gsii.inference import Inference
    
    setup_logging(logfname='logs/generate.log', level=WARN)
    device     = 'cpu'
    model_dir  = '../pretrained_model_indonesia'
    model_fn   = 'epoch140.pt'
    data_dir   = '../data/AMR/amr_silver_indonesia'
    test_data  = 'test.txt.features'
    out_fn     = model_fn + '.test_generated_silver'

    infer = Inference(model_dir, model_fn, device=device)
    infer.reparse_annotated_file(data_dir, test_data, model_dir, out_fn)
