import pickle
import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


def parse_data(x, attributes):
    x = x.set_index("Parameter").to_dict()["Value"]

    values = []

    for attr in attributes:
        if x.__contains__(attr):
            values.append(x[attr])
        else:
            values.append(np.nan)
    return values


def parse_id(id_, missing_ratio=0.1, attributes=None, inputdir="", mask=None):
    data = pd.read_csv(inputdir + "/{}.txt".format(id_))

    mask_tmp = mask.loc[id_]

    # create data for timpoints x attributes
    observed_values = []
    for h in range(8):
        observed_values.append(parse_data(data[data["Time"] == h], attributes))
    observed_values = np.array(observed_values)
    observed_masks = ~np.isnan(observed_values)

    gt_masks = observed_masks.copy()
    gt_masks[mask_tmp==0] = False

    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")

    return observed_values, observed_masks, gt_masks


def get_idlist(inputdir=""):
    patient_id = []
    for filename in os.listdir(inputdir):
        match = re.search(".*.txt", filename)

        if match:
            patient_id.append(match.group().split('.txt')[0])
    patient_id = np.sort(patient_id)
    return patient_id


class vatanen_Dataset(Dataset):
    def __init__(self, eval_length=6, use_index_list=None, missing_ratio=0.0, seed=0, attributes=None, inputdir=""):
        self.eval_length = eval_length
        np.random.seed(seed)

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []
        path = (
            "./data/vatanen_MR-" + str(missing_ratio) + ".pk"
        )

        mask = pd.read_csv("../indata/mask_" + str(missing_ratio) + ".csv", index_col=0)
        if os.path.isfile(path) == False:  # if datasetfile is none, create
            idlist = get_idlist(inputdir=inputdir)
            for id_ in idlist:
                try:
                    observed_values, observed_masks, gt_masks = parse_id(
                        id_, missing_ratio, attributes, inputdir, mask
                    )
                    self.observed_values.append(observed_values)
                    self.observed_masks.append(observed_masks)
                    self.gt_masks.append(gt_masks)
                except Exception as e:
                    print(id_, e)
                    continue
            self.observed_values = np.array(self.observed_values)
            self.observed_masks = np.array(self.observed_masks)
            self.gt_masks = np.array(self.gt_masks)

            with open(path, "wb") as f:
                pickle.dump(
                    [self.observed_values, self.observed_masks, self.gt_masks], f
                )
        else:  # load datasetfile
            with open(path, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks = pickle.load(
                    f
                )
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, batch_size=16, missing_ratio=0.1, inputdir="", attributes=None, foldername="", train_index=None, test_index=None):

    np.random.seed(seed)
    np.random.shuffle(train_index)
    num_train = (int)(len(train_index) * 0.9)
    tmp_index = train_index
    train_index = tmp_index[:num_train]
    valid_index = tmp_index[num_train:]

    dataset = vatanen_Dataset(
        eval_length=8, use_index_list=train_index, missing_ratio=missing_ratio, seed=seed, attributes=attributes
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = vatanen_Dataset(
        eval_length=8, use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = vatanen_Dataset(
        eval_length=8, use_index_list=test_index, missing_ratio=missing_ratio, seed=seed
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    
    return train_loader, valid_loader, test_loader




