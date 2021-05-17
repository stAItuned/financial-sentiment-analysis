import unittest

from constants.config import MOVIE_DATASET, TOKENIZER, SMOTE_IMBALANCE
from scripts.pipelines.preprocessing_pipeline import preprocessing_pipeline


class Vectorization_Test(unittest.TestCase):

    def test_tokenizing(self):
        seed = 2021

        emb_dim = 100

        dataset_params = {'data_path': 'resources/preprocessed_data/cleaned_data_v1.csv',
                          'dataset_type': MOVIE_DATASET,
                          'preprocessed': True,
                          # 'target_scaling': (0, 1),
                          'vectorization': TOKENIZER,
                          'imbalance': None,
                          'imb_params': {'random_state': seed,
                                         'k_neighbors': 3},
                          'train': True}

        x, y, dataset = preprocessing_pipeline(dataset_params)

        self.assertEqual(len(x), len(y))


if __name__ == '__main__':
    unittest.main()
