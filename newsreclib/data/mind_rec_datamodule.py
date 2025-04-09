from typing import Any, Dict, List, Optional
import torch
import os
import hydra

import torch.nn as nn
from lightning import LightningDataModule
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from newsreclib.data.components.mind_dataframe import MINDDataFrame
from newsreclib.data.components.rec_dataset import (
    DatasetCollate,
    RecommendationDatasetTest,
    RecommendationDatasetTrain,
)


class MINDRecDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_size: str,
        dataset_url: Dict[str, Dict[str, str]],
        data_dir: str,
        dataset_attributes: List[str],
        id2index_filenames: Dict[str, str],
        entity_embeddings_filename: str,
        use_plm: bool,
        entity_embed_dim: int,
        entity_freq_threshold: int,
        entity_conf_threshold: float,
        valid_time_split: str,
        max_title_len: int,
        max_abstract_len: int,
        max_history_len: int,
        neg_sampling_ratio: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        drop_last: bool,
        sentiment_annotator: Optional[nn.Module] = None,
    ):
        """Initialize a MINDRecDataModule.
        
        Removed GloVe-related parameters:
        - pretrained_embeddings_url
        - word_embeddings_dirname
        - word_embeddings_fpath
        - use_pretrained_categ_embeddings
        - word_embed_dim
        - categ_embed_dim
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Core parameters
        self.dataset_size = dataset_size
        self.dataset_url = dataset_url
        self.data_dir = data_dir
        self.dataset_attributes = dataset_attributes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def _load_news_ids(self):
        news_ids = {}
        news_path = os.path.join(self.hparams.data_dir, "news.tsv")
        with open(news_path, 'r', encoding='utf-8') as f:
            for line in f:
                news_id = line.split('\t')[0]
                news_ids[news_id] = len(news_ids)
        return news_ids
    
    def prepare_data(self):
        """Download data if needed."""
        # download train set
        MINDDataFrame(
            dataset_size=self.hparams.dataset_size,
            dataset_url=self.hparams.dataset_url,
            data_dir=self.hparams.data_dir,
            dataset_attributes=self.hparams.dataset_attributes,
            custom_embedding_path=self.hparams.custom_embedding_path,
            valid_time_split=self.hparams.valid_time_split,
            train=True,
            validation=False,
            download=True,
            # Remove any GloVe-related parameters
        )

        # Similar changes for validation and test set initializations

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = MINDDataFrame(
                dataset_size=self.hparams.dataset_size,
                dataset_url=self.hparams.dataset_url,
                data_dir=self.hparams.data_dir,
                dataset_attributes=self.hparams.dataset_attributes,
                custom_embedding_path=self.hparams.custom_embedding_path,
                valid_time_split=self.hparams.valid_time_split,
                train=True,
                validation=False,
                download=False,
            )
            validset = MINDDataFrame(
                dataset_size=self.hparams.dataset_size,
                dataset_url=self.hparams.dataset_url,
                data_dir=self.hparams.data_dir,
                dataset_attributes=self.hparams.dataset_attributes,
                custom_embedding_path=self.hparams.custom_embedding_path,
                valid_time_split=self.hparams.valid_time_split,
                train=True,
                validation=True,
                download=False,
            )
            testset = MINDDataFrame(
                dataset_size=self.hparams.dataset_size,
                dataset_url=self.hparams.dataset_url,
                data_dir=self.hparams.data_dir,
                dataset_attributes=self.hparams.dataset_attributes,
                custom_embedding_path=self.hparams.custom_embedding_path,
                valid_time_split=self.hparams.valid_time_split,
                train=False,
                validation=False,
                download=False,
            )

            self.data_train = RecommendationDatasetTrain(
                news=trainset.news,
                behaviors=trainset.behaviors,
                max_history_len=self.hparams.max_history_len,
                neg_sampling_ratio=self.hparams.neg_sampling_ratio,
            )
            self.data_val = RecommendationDatasetTest(
                news=validset.news,
                behaviors=validset.behaviors,
                max_history_len=self.hparams.max_history_len,
            )
            self.data_test = RecommendationDatasetTest(
                news=testset.news,
                behaviors=testset.behaviors,
                max_history_len=self.hparams.max_history_len,
            )
        
        super().setup(stage)

        # Explicitly take only first 100 samples
        self.trainset.behaviors = self.trainset.behaviors.head(100)
        self.valset.behaviors = self.valset.behaviors.head(100)
        self.testset.behaviors = self.testset.behaviors.head(100)

        print("Explicitly running with first 100 samples only.")

        if self.custom_embeddings:
            # Validate embeddings explicitly match news_ids
            news_ids = self.trainset.news["news_id"].tolist()
            assert set(news_ids).issubset(set(self.custom_embeddings.keys())), \
                "Embedding keys mismatch with news IDs."


    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            collate_fn=DatasetCollate(
                dataset_attributes=self.hparams.dataset_attributes,
                use_plm=self.hparams.use_plm,
                tokenizer=self.tokenizer if self.hparams.use_plm else None,
                max_title_len=self.hparams.max_title_len if not self.hparams.use_plm else None,
                max_abstract_len=self.hparams.max_abstract_len,
                concatenate_inputs=self.hparams.concatenate_inputs,
            ),
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            collate_fn=DatasetCollate(
                dataset_attributes=self.hparams.dataset_attributes,
                use_plm=self.hparams.use_plm,
                tokenizer=self.tokenizer if self.hparams.use_plm else None,
                max_title_len=self.hparams.max_title_len,
                max_abstract_len=self.hparams.max_abstract_len,
                concatenate_inputs=self.hparams.concatenate_inputs,
            ),
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            collate_fn=DatasetCollate(
                dataset_attributes=self.hparams.dataset_attributes,
                use_plm=self.hparams.use_plm,
                tokenizer=self.tokenizer if self.hparams.use_plm else None,
                max_title_len=self.hparams.max_title_len if not self.hparams.use_plm else None,
                max_abstract_len=self.hparams.max_abstract_len,
                concatenate_inputs=self.hparams.concatenate_inputs,
            ),
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
