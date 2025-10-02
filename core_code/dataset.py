import pickle
import torch
from torch.utils.data import Dataset

class MYDataset(Dataset):
    def __init__(self,d1_file,gram_data_file,label_file):
        with open(d1_file, "rb") as f:
            self.molvec= pickle.load(f)
        with open(gram_data_file, "rb") as f:
            self.gram = pickle.load(f)
        with open(label_file, "rb") as f:
            self.y_list = pickle.load(f)
        self.pos_num = sum(1 for label in self.y_list if label == 1)

    def __getitem__(self, index):
        data2=self.gram[index]
        #data的返回值需要是tensor类型
        data2=torch.Tensor(data2)
        data2 = data2.unsqueeze(0)

        data1=self.molvec[index]
        data1 = torch.Tensor(data1)
        data1 = data1.unsqueeze(0)

        # data=torch.cat((data1,data2),-2)
        label=self.y_list[index]
        return data1,data2,label

    #这个方法返回数据集长度
    def __len__(self):
        return len(self.y_list)

    def get_positive_count(self):
        """返回正样本数量"""
        return self.pos_num


class batch_infer_Dataset(Dataset):
    def __init__(self,vec1,vec2):
       self.molvec=vec1
       self.gram=vec2
    def __getitem__(self, index):
        data2=self.gram[index]
        #data的返回值需要是tensor类型
        data2=torch.Tensor(data2)
        data2 = data2.unsqueeze(0)

        data1=self.molvec[index]
        data1 = torch.Tensor(data1)
        data1 = data1.unsqueeze(0)

        # data=torch.cat((data1,data2),-2)
        return data1,data2

    def __len__(self):
        return len(self.molvec)
