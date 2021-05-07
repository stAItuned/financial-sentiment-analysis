import unittest
from constants.paths import PREPROCESSED_DATA_DIR
from scripts.data.extraction import extract_dataset
from scripts.data.preprocessing import data_preprocessing
from scripts.pipelines.preprocessing_pipeline import preprocessing_oversampling_tdidf


class DataTests(unittest.TestCase):

    def test_load_txt_data(self):
        data = extract_dataset('../resources/FinancialPhraseBank/Sentences_AllAgree.txt')
        data = extract_dataset('../resources/FinancialPhraseBank/Sentences_50Agree.txt')
        data = extract_dataset('../resources/FinancialPhraseBank/Sentences_66Agree.txt')
        data = extract_dataset('../resources/FinancialPhraseBank/Sentences_75Agree.txt')


    def test_preprocessing(self):
        data = extract_dataset('../resources/kaggle/train.tsv')
        prep_data = data_preprocessing(data,
                                       # save_dir=PREPROCESSED_DATA_DIR
                                       )
        return prep_data

    def test_preprocessing_oversampling(self):
        preprocessing_oversampling_tdidf('../resources/preprocessed_data/cleaned_data_v1.csv')


if __name__ == '__main__':
    unittest.main()
