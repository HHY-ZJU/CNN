import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np

class DatasetProcessing(Dataset):
        def __init__(self, data_path, img_label_path, transform=None):
            self.img_path = data_path
            self.transform = transform

            file = os.listdir(img_label_path)
            img_filename0 = []
            for x in file:
                tmp_name = x.split('_')
                if len(tmp_name) > 3:
                    img_filename0.append(x.split('.txt')[0])
                else:
                    img_filename0.append(x.split('.txt')[0] + '.jpg')
            self.img_filename = img_filename0
            # self.img_filename = [x.split('.txt')[0] + '.jpg' for x in file]

            label = []
            for each in file:
                fp = open(os.path.join(img_label_path, each), 'r')
                f = fp.readlines()
                lb = np.zeros([25], dtype=np.int64)
                for each0 in f:
                    index = each0.split('\n')[0]
                    lb[int(index)] = 1
                label.append(lb)
                fp.close()
            label = np.array(label, dtype=np.int64)
            # labels = np.loadtxt(label_filepath, dtype=np.int64)
            self.label = label

        def __getitem__(self, index):

            img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
            img = img.convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            label = torch.from_numpy(self.label[index])
            return img, label

        def __len__(self):
            return len(self.img_filename)

