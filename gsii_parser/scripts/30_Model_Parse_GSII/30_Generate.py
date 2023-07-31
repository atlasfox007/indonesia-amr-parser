if __name__ == '__main__':
    import os
    import setup_run_dir    # this import tricks script to run from 2 levels up
    from   amrlib.utils.logging import setup_logging, WARN
    from   amrlib.models.parse_gsii.inference import Inference
    
    setup_logging(logfname='logs/generate.log', level=WARN)
    device     = 'cuda:0'
    model_dir  = '../pretrained_model'
    model_fn   = 'epoch15.pt'
    data_dir   = '../test_input'
    test_data  = 'test_input.txt.raw.features'
    out_fn     = model_fn + '.test_generated'

    infer = Inference(model_dir, model_fn, device=device)
    infer.reparse_annotated_file(data_dir, test_data, model_dir, out_fn)
