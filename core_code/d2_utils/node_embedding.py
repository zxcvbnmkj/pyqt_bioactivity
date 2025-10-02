"""该文件是 pytorch 框架完整的模型训练步骤，负责训练节点嵌入模型Cbow"""
from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np

# 定义模型结构，需要训练
class CBoW(nn.Module):
    def __init__(self, feature_num, embedding_dim, task_num, task_size_list):
        super(CBoW, self).__init__()
        self.task_num = task_num
        self.embeddings = nn.Linear(feature_num, embedding_dim, bias=False)
        self.layers = nn.ModuleList()
        for task_size in task_size_list:
            self.layers.append(nn.Sequential(
                nn.Linear(embedding_dim, 20),
                nn.ReLU(),
                nn.Linear(20, len(task_size)),
            ))

    def forward(self, x):
        embeds = self.embeddings(x)
        embeds = embeds.sum(1)

        outputs = []
        for layer in self.layers:
            output = layer(embeds)
            outputs.append(output)
        return outputs

# 定义的数据读取类
class GraphDataset(Dataset):
    def __init__(self,data_path,padding_size,max_atoms):
        self.X_data, self.Y_label_list = [], []
        X_data, Y_label_list = get_data(data_path=data_path, padding_size=padding_size,max_atoms=max_atoms)
        self.X_data.extend(X_data)
        self.Y_label_list.extend(Y_label_list)
        self.X_data = np.array(self.X_data)
        self.Y_label_list = np.array(self.Y_label_list)
        print('data size: ', self.X_data.shape, '\tlabel size: ', self.Y_label_list.shape)
        print(len(self.X_data), "X_data的长度")

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        x_data = self.X_data[idx]
        y_label_list = self.Y_label_list[idx]

        x_data = torch.from_numpy(x_data)
        y_label_list = torch.from_numpy(y_label_list)
        return x_data, y_label_list

def get_data(data_path, padding_size,max_atoms):
    FEATURE_NUM=42
    segmentation_list = [range(0, 10), range(10, 17), range(17, 24), range(24, 30), range(30, 36),
                         range(36, 38), range(38, 40), range(40, 42)]
    data = np.load(data_path)
    print(data.keys())
    print(data_path)
    adjacent_matrix_list = data['adjacent_matrix_list']
    node_attribute_matrix_list = data['node_attribute_matrix_list']

    molecule_num = adjacent_matrix_list.shape[0]
    print('molecule num\t', molecule_num)

    X_data = []
    Y_label_list = []

    print('adjacent_matrix_list shape: {}\tnode_attribute_matrix_list shape: {}'.format(adjacent_matrix_list.shape, node_attribute_matrix_list.shape))

    for adjacent_matrix, node_attribute_matrix in zip(adjacent_matrix_list, node_attribute_matrix_list):
        assert len(adjacent_matrix) == max_atoms
        assert len(node_attribute_matrix) == max_atoms
        for i in range(max_atoms):
            if sum(adjacent_matrix[i]) == 0:
                break
            x_temp = np.zeros((padding_size, FEATURE_NUM))
            cnt = 0
            for j in range(max_atoms):
                if adjacent_matrix[i][j] == 1:
                    x_temp[cnt] = node_attribute_matrix[j]
                    cnt += 1
            x_temp = np.array(x_temp)

            y_temp = []
            atom_feat = node_attribute_matrix[i]
            for s in segmentation_list:
                y_temp.append(atom_feat[s].argmax())

            X_data.append(x_temp)
            Y_label_list.append(y_temp)

    X_data = np.array(X_data)
    Y_label_list = np.array(Y_label_list)
    return X_data, Y_label_list





def train(model,train_dataloader,optimizer,epochs,weight_file):
    criterion = nn.CrossEntropyLoss()
    model.train()

    optimal_loss = 1e7
    for epoch in range(epochs):
        train_loss = []

        for batch_id, (x_data, y_actual) in enumerate(train_dataloader):
            x_data = Variable(x_data).float()
            y_actual = Variable(y_actual).long()
            # if torch.cuda.is_available():
            #     x_data = x_data.cuda()
            #     y_actual = y_actual.cuda()
            optimizer.zero_grad()
            y_predict = model(x_data)

            loss = 0
            for i in range(8):
                y_true, y_pred = y_actual[..., i], y_predict[i]
                temp_loss = criterion(y_pred, y_true)
                loss += temp_loss
            loss.backward()
            optimizer.step()
            #pytorch0.4之后，不需要加[]了
            #train_loss += loss.data[0] 是pytorch0.3.1版本代码,在0.4-0.5版本的pytorch会出现警告,不会报错,但是0.5版本以上的pytorch就会报错,总的来说是版本更新问题.
            #train_loss+=loss.data[0]改为train_loss+=loss.item()
            train_loss.append(loss.item())
            # train_loss.append(loss.data[0])

        train_loss = np.mean(train_loss)
        print('epoch: {}\tloss is: {}'.format(epoch, train_loss))
        if train_loss < optimal_loss:
            optimal_loss = train_loss
            print('Saving model at epoch {}\toptimal loss is {}.'.format(epoch, optimal_loss))
            torch.save(model.state_dict(), weight_file)


def test(dataloader,model):
    model.eval()
    accuracy, total = 0, 0
    for batch_id, (x_data, y_actual) in enumerate(dataloader):
        x_data = Variable(x_data).float()
        y_actual = Variable(y_actual).long()
        # if torch.cuda.is_available():
        #     x_data = x_data.cuda()
        #     y_actual = y_actual.cuda()
        y_predict = model(x_data)
        for i in range(8):
            y_true, y_pred = y_actual[..., i].cpu().data.numpy(), y_predict[i].cpu().data.numpy()
            y_pred = y_pred.argmax(1)
            accuracy += np.sum(y_true == y_pred)
            total += y_pred.shape[0]
    accuracy = 1. * accuracy / total
    print('Accuracy: {}'.format(accuracy))

# 执行训练节点嵌入函数 Cbow 的主函数
# 输入：训练数据路径、验证数据路径
def node_embedding(train_path,test_path,weight_path,max_atoms):
    epochs = 20
    random_dimension = 50
    segmentation_num = 8
    padding_size = 10

    segmentation_list = [range(0, 10), range(10, 17), range(17, 24), range(24, 30), range(30, 36),
                         range(36, 38), range(38, 40), range(40, 42)]
    #创建模型实例
    model = CBoW(feature_num=42, embedding_dim=random_dimension,
                 task_num=segmentation_num, task_size_list=segmentation_list)

    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)

    train_dataset = GraphDataset(train_path,padding_size=padding_size,max_atoms=max_atoms)
    test_dataset = GraphDataset(test_path, padding_size=padding_size,max_atoms=max_atoms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)

    train(model,train_dataloader,optimizer,epochs,weight_path)

    test(train_dataloader,model)
    test(test_dataloader,model)

if __name__ == '__main__':
    node_embedding("./tmp/X_train_d2feature.npz",
                   "./tmp/X_test_d2feature.npz",
                   weight_path="./tmp/CBoW_50dim.pt", max_atoms=100)