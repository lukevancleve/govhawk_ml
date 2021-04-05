import logging
import time
import pickle

from pandarallel import pandarallel

from src.models.data_loader import createDeepLegisDataFrame
from src.models.configurationClasses import deepLegisConfig


def create_pretokenized_dataset():

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Use every core on the machine.
    pandarallel.initialize(use_memory_fs=False)


    config = deepLegisConfig("bert_128.json")

    # Create a dataframe out of the ml_data.csv by adding the text to it.
    df, _ = createDeepLegisDataFrame(config, read_cached=False)

    # Take the text and tokenize it into the final product the model wants to see.
    tokenizer = config.tokenizer
    def tokenizer_wrapper(text):
        d = tokenizer(text, truncation=True, padding='max_length', max_length=config.max_length)
        return(d['input_ids'])

    tic = time.perf_counter()
    df['tokens'] = df.text.parallel_apply( tokenizer_wrapper)
    toc = time.perf_counter()

    logger.info(f"Tokenized in {(toc-tic)/60.0} min -  {toc - tic:0.4f} seconds")

    print(df.head())

    # Save it for later use
    pickle_file = config.data_vol + "preprocessed_df_128.pkl"
    pickle.dump(df, open(pickle_file, "wb" ))

if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    create_pretokenized_dataset()