from nlp_id import Tokenizer
import multiprocessing
from tqdm import tqdm

data_chunks = ["Halo nama saya Joko Widodo, tinggal di Jakarta Barat!",
               "Halo nama saya Joko Widodo, tinggal di Jakarta Barat!",
               "Halo nama saya Joko Widodo, tinggal di Jakarta Barat!",
               "Halo nama saya Joko Widodo, tinggal di Jakarta Barat!",
               "Halo nama saya Joko Widodo, tinggal di Jakarta Barat!",
               "Halo nama saya Joko Widodo, tinggal di Jakarta Barat!",
               "Halo nama saya Joko Widodo, tinggal di Jakarta Barat!",
               "Halo nama saya Joko Widodo, tinggal di Jakarta Barat!",
               "Halo nama saya Joko Widodo, tinggal di Jakarta Barat!",
               "Halo nama saya Joko Widodo, tinggal di Jakarta Barat!",
               "Halo nama saya Joko Widodo, tinggal di Jakarta Barat!",
               "Halo nama saya Joko Widodo, tinggal di Jakarta Barat!"]

loaded_model : Tokenizer = None
def load_model():
    global loaded_model
    if loaded_model is not None:
        return
    import pickle
    with open('tokenizer_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

def process_data(data) :
    result = loaded_model.tokenize(data)
    return result

def do():
    load_model()

    try:
        # Process the data in parallel using the pool and the `process_data` function
        with multiprocessing.Pool(processes=4) as pool:
            for data in tqdm(pool.imap(process_data, data_chunks), total= len(data_chunks)):
                continue

    except Exception as e:
        print("Multiprocessing with the loaded model failed:", e)
    else:
        print("Multiprocessing with the loaded model successful!")
        
    finally:
        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()