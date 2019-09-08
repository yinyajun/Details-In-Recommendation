#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/18 15:14
# @Author  : Yajun Yin
# @Note    :

import scipy.sparse
from convert import *

from mlperf_compliance import mlperf_log

TRAIN_DATASET_FILENAME = 'train_dataset.csv'


class CFTrainDataset(object):
    def __init__(self, train_fname, nb_neg):
        self._load_train_matrix(train_fname)
        self.nb_neg = nb_neg

        mlperf_log.ncf_print(key=mlperf_log.INPUT_HP_SAMPLE_TRAIN_REPLACEMENT)

    def _load_train_matrix(self, train_fname):
        def process_line(line):
            tmp = line.split('\t')
            return [int(tmp[0]), int(tmp[1]), float(tmp[2]) > 0]

        with open(train_fname, 'r') as file:
            data = list(map(process_line, file))
        self.nb_users = max(data, key=lambda x: x[0])[0] + 1
        self.nb_items = max(data, key=lambda x: x[1])[1] + 1

        self.data = list(filter(lambda x: x[2], data))  # filter x[2] == False, not necessary
        self.mat = scipy.sparse.dok_matrix(
            (self.nb_users, self.nb_items), dtype=np.float32)
        for user, item, _ in data:
            self.mat[user, item] = 1.

    def __len__(self):
        return (self.nb_neg + 1) * len(self.data)

    def __getitem__(self, idx):
        # positive example
        if idx % (self.nb_neg + 1) == 0:
            idx = idx // (self.nb_neg + 1)
            return self.data[idx][0], self.data[idx][1], 1
            # negative example
        else:
            idx = idx // (self.nb_neg + 1)
            u = self.data[idx][0]
            j = np.random.choice(self.nb_items, 1).tolist()[0]
            while (u, j) in self.mat:
                j = np.random.choice(self.nb_items, 1).tolist()[0]
            return u, j, 0


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--output', type=str, default='./',
                        help='Output directory for train dataset CSV files')
    parser.add_argument('-n', '--nb_neg', type=int, default=4,
                        help='Number of negative samples for each positive'
                             'train example')
    return parser.parse_args()


def _train_generator(train_fname, nb_neg):
    df = CFTrainDataset(train_fname, nb_neg)
    print("Generating {} negative samples for each example in train CSV files".format(nb_neg))
    for row in tqdm(range(len(df))):
        yield df[row]


def main():
    args = parse_args()
    # generate train examples and save.
    mlperf_log.ncf_print(key=mlperf_log.INPUT_STEP_TRAIN_NEG_GEN, value=args.nb_neg)
    train_generator = _train_generator(os.path.join(args.output, TRAIN_RATINGS_FILENAME), args.nb_neg)
    train_df = pd.DataFrame(train_generator)
    train_df.to_csv(os.path.join(args.output, TRAIN_DATASET_FILENAME), index=False, header=False, sep='\t')


if __name__ == '__main__':
    main()
