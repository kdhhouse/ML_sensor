import os
import csv

import torch
from torch.utils.data import Dataset

import numpy as np


class DatasetLoader(Dataset):
    def __init__(self, input_dir, num):
        self.input_dir = input_dir

        self.num = num

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        f = open(self.input_dir, "r", encoding="utf-8")
        rdr = csv.reader(f)

        input_data = []
        label_data = []

        for line in rdr:
            count = 0
            input_content = []
            label_content = []
            for content in line:
                if 21 < count < 28:
                    input_content.append(float(content)*100)
                elif count < 6:
                    label_content.append(float(content)*0.1)

                else:
                    pass
                count += 1
            input_data.append(input_content)
            label_data.append(label_content)

        input_data = np.array(input_data)
        # input_data = np.array(input_data) + 0.02*np.random.rand(np.shape(input_data)[0],np.shape(input_data)[1])
        label_data = np.array(label_data)*[1, 1, 1, 0.02, 0.02, 0.02]*1

        f.close()
        return {
            "input_data": torch.from_numpy(input_data).type(torch.FloatTensor),
            "label_data": torch.from_numpy(label_data).type(torch.FloatTensor),
        }
