if __name__ == '__main__':
    import os
    import setup_run_dir    # this import tricks script to run from 2 levels up
    from   src.utils.logging import setup_logging, WARN
    from   src.models.parse_gsii.inference import Inference
    
    setup_logging(logfname='logs/generate.log', level=WARN)
    device     = 'cuda:0'
    model_dir  = '../pretrained_model_indonesia'
    model_fn   = 'epoch140.pt'
    data_dir   = '../data/AMR/amr_gold_indonesia'
    test_data  = 'amr_simple_test.txt.features'
    out_fn     = model_fn + '.test_generated_singlesentence'

    infer = Inference(model_dir, model_fn, device=device)
    l = infer.parse_sents(["Halo nama saya adalah Bima"])
    # infer.reparse_annotated_file(data_dir, test_data, model_dir, out_fn)
    print(l[0])