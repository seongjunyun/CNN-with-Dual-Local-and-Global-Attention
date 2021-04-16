import pickle

import numpy as np
import pandas as pd
import torch
from nltk import word_tokenize
from torch.utils import data
from torch.utils.data.dataloader import default_collate


class ReviewDataset(data.Dataset):

    def __init__(self, csv_path, root='data/', max_len=10000):

        self.dataset = pd.read_csv(root + csv_path)

        # load glove
        glove_path = root + 'glove.pickle'
        with open(glove_path, 'rb') as f:
            self.glove = pickle.load(f)

        self.temp = np.random.uniform(0, 1, 100)
        self.pad = np.zeros(100)
        self.unknown = np.random.uniform(0, 1, 100)
        self.delimiter = np.random.uniform(0, 1, 100)
        self.max_len = max_len

    def __getitem__(self, index):

        row = self.dataset.loc[index]

        user_id = row['user_id']
        item_id = row['item_id']

        # user review
        user_path = "data/User/" + user_id + ".tsv"
        user_review = self.preprocess_review(user_path)

        # item review

        item_path = "data/Item/" + item_id + ".tsv"
        item_review = self.preprocess_review(item_path)

        # rating

        target = torch.Tensor([row['rating']]).float()

        if user_review is not None:

            user_review = torch.from_numpy(user_review).float()

        else:
            return None

        if item_review is not None:

            item_review = torch.from_numpy(item_review).float()

        else:
            return None

        return (user_review, item_review, target)

    def __len__(self):
        """Returns the total number of image files."""
        return len(self.dataset)

    def preprocess_review(self, file_path):

        try:
            reviews = pd.read_csv(file_path, sep='\t')
        except Exception as e:
            return None

        review_len = len(reviews.keys())

        reviews = reviews.keys()
        if review_len > 100:
            reviews = reviews[:100]

        total_review = np.array([])
        for review_str in reviews:
            review = word_tokenize(review_str)
            total_review = np.concatenate((total_review, review))
            total_review = np.append(total_review, '+++')

        review = []
        for word in total_review:
            if word == '+++':
                review.append(self.delimiter)
            else:
                if word in self.glove:
                    review.append(self.glove[word])
                else:
                    review.append(self.unknown)

        review = np.array(review)

        if len(review) < self.max_len:
            pad_len = self.max_len - len(review)

            pad_vector = np.zeros((pad_len, 100))
            review = np.concatenate((review, pad_vector), axis=0)
        else:
            review = review[:self.max_len]

        return review


def my_collate(batch):
    batch = filter(lambda x: x is not None, batch)
    return default_collate(batch)


def get_loader(data_path, batch_size=100, shuffle=True, num_workers=2):
    """Builds and returns Dataloader."""

    dataset = ReviewDataset(data_path)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  collate_fn=my_collate)
    return data_loader
