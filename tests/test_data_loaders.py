import unittest
from src.models.configurationClasses import  deepLegisConfig
from src.models.data_loader import *
import os


class TestDataCode(unittest.TestCase):

    def test_base_class(self):

        with self.assertRaises(TypeError):
            legislationDataset() # Abstract class
    
    def test_loading_df(self):

        config = deepLegisConfig('distilbert_128.json')

        df, encoder = createDeepLegisDataFrame(config, reduce_by_factor=1000)
        cols = df.columns
        #print(df.head())
        expected_columns = ['passed', 'signed', 'text', 'id', 'version_number', \
                            'bill_id', 'partisan_lean', 'sc_id_cat', 'sc_id']
        for col in cols:
            self.assertIn(col, expected_columns)

    def test_no_missing(self):

        config = deepLegisConfig('distilbert_128.json')
        df, encoder = createDeepLegisDataFrame(config,)
        for col in df.columns:
            print("testing no missing in:" + col)
            self.assertEqual(sum(df[col].isna()), 0)


    def test_pl_batches(self):

        config = deepLegisConfig('distilbert_128.json')
        text_only_dataset = legislationDatasetPartisanLean(config)
        df, encoder = createDeepLegisDataFrame(config, reduce_by_factor=1000)

        train_data, val_data, test_data, full_data, split_data = \
            text_only_dataset.create_batch_stream(df)

        for elem in train_data.take(1):
            x, y = elem
            self.assertEqual(x['input_ids'].shape,  (config.batch_size, config.max_length ))
            self.assertEqual(x['partisan_lean'].shape,  (config.batch_size,  ))
            self.assertEqual(y.shape, (config.batch_size,))


    def test_text_only_batches(self):

        config = deepLegisConfig('distilbert_128.json')
        text_only_dataset = legislationDatasetText(config)
        df, encoder = createDeepLegisDataFrame(config, reduce_by_factor=1000)

        train_data, val_data, test_data, full_data, split_data = \
            text_only_dataset.create_batch_stream(df)

        for elem in train_data.take(1):
            x, y = elem
            self.assertEqual(x['input_ids'].shape,  (config.batch_size, config.max_length ))
            self.assertEqual(y.shape, (config.batch_size,))


    def test_all_batches(self):

        config = deepLegisConfig('distilbert_128.json')
        all_dataset = legislationDatasetAll(config)
        df, encoder = createDeepLegisDataFrame(config, reduce_by_factor=1000)
        all_dataset.config.n_sc_id_classes = len(encoder.classes_)

        train_data, val_data, test_data, full_data, split_data = \
            all_dataset.create_batch_stream(df)

        for elem in train_data.take(1):
            x, y = elem
        self.assertEqual(x['input_ids'].shape,  (config.batch_size, config.max_length ))
        self.assertEqual(x['version_number'].shape,  (config.batch_size,  ))
        self.assertEqual(x['partisan_lean'].shape,  (config.batch_size,  ))
        self.assertEqual(x['sc_id'].shape,  (config.batch_size, len(encoder.classes_)))
        self.assertEqual(y.shape, (config.batch_size,))


    def test_no_text_batches(self):

        config = deepLegisConfig('no_text.json')
        all_dataset = legislationDatasetNoText(config)
        df, encoder = createDeepLegisDataFrame(config, reduce_by_factor=1000)
        all_dataset.config.n_sc_id_classes = len(encoder.classes_)

        train_data, val_data, test_data, full_data, split_data = \
            all_dataset.create_batch_stream(df)

        for elem in train_data.take(1):
            x, y = elem
        self.assertEqual(x['version_number'].shape,  (config.batch_size,  ))
        self.assertEqual(x['partisan_lean'].shape,  (config.batch_size,  ))
        self.assertEqual(x['sc_id'].shape,  (config.batch_size, len(encoder.classes_)))
        self.assertEqual(y.shape, (config.batch_size,))



    def test_rev_cat_batches(self):

        config = deepLegisConfig('distilbert_128.json')
        all_dataset = legislationDatasetRevCat(config)
        df, encoder = createDeepLegisDataFrame(config, reduce_by_factor=1000)
        all_dataset.config.n_sc_id_classes = len(encoder.classes_)

        train_data, val_data, test_data, full_data, split_data = \
            all_dataset.create_batch_stream(df)

        for elem in train_data.take(1):
            x, y = elem
        self.assertEqual(x['input_ids'].shape,  (config.batch_size, config.max_length ))
        self.assertEqual(x['version_number'].shape,  (config.batch_size,  ))
        self.assertEqual(x['sc_id'].shape,  (config.batch_size, len(encoder.classes_)))
        self.assertEqual(y.shape, (config.batch_size,))


    def test_new_tokenizer_length(self):

        config = deepLegisConfig('distilbert_128.json')
        config.max_length = 512
        text_only_dataset = legislationDatasetPartisanLean(config)
        df, encoder = createDeepLegisDataFrame(config, reduce_by_factor=1000)

        train_data, val_data, test_data, full_data, split_data = \
            text_only_dataset.create_batch_stream(df)

        for elem in train_data.take(1):
            x, y = elem
            self.assertEqual(x['input_ids'].shape,  (config.batch_size, config.max_length ))
            self.assertEqual(x['partisan_lean'].shape,  (config.batch_size,  ))
            self.assertEqual(y.shape, (config.batch_size,))



if __name__ == '__main__':

    unittest.main()