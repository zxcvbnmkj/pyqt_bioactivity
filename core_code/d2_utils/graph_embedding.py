from __future__ import print_function
import pickle
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np

from core_code.d2_utils.tools import extract_feature_and_label_npy, get_resource_path
from core_code.d2_utils.node_embedding import CBoW


def get_data(data_path):
    data = np.load(data_path)
    print(data.keys())
    adjacent_matrix_list = data['adjacent_matrix_list']
    distance_matrix_list = data['distance_matrix_list']
    bond_attribute_matrix_list = data['bond_attribute_matrix_list']
    node_attribute_matrix_list = data['node_attribute_matrix_list']
    kwargs = {}
    return adjacent_matrix_list, distance_matrix_list, bond_attribute_matrix_list,\
           node_attribute_matrix_list, kwargs

class GraphDataset(Dataset):
    def __init__(self, node_attribute_matrix_list, adjacent_matrix_list, distance_matrix_list):
        self.node_attribute_matrix_list = node_attribute_matrix_list
        self.adjacent_matrix_list = adjacent_matrix_list
        self.distance_matrix_list = distance_matrix_list

    def __len__(self):
        return len(self.node_attribute_matrix_list)

    def __getitem__(self, idx):
        node_attribute_matrix = torch.from_numpy(self.node_attribute_matrix_list[idx])
        adjacent_matrix = torch.from_numpy(self.adjacent_matrix_list[idx])
        distance_matrix = torch.from_numpy(self.distance_matrix_list[idx])
        return node_attribute_matrix, adjacent_matrix, distance_matrix


def get_walk_representation(dataloader,model):
    X_embed = []
    embedded_graph_matrix_list = []
    for batch_id, (node_attribute_matrix, adjacent_matrix, distance_matrix) in enumerate(dataloader):
        node_attribute_matrix = Variable(node_attribute_matrix).float()
        adjacent_matrix = Variable(adjacent_matrix).float()
        distance_matrix = Variable(distance_matrix).float()
        # if torch.cuda.is_available():
        #     node_attribute_matrix = node_attribute_matrix.cuda()
        #     adjacent_matrix = adjacent_matrix.cuda()
        #     distance_matrix = distance_matrix.cuda()

        tilde_node_attribute_matrix = model.embeddings(node_attribute_matrix)

        walk = tilde_node_attribute_matrix
        v1 = torch.sum(walk, dim=1)

        walk = torch.bmm(adjacent_matrix, walk) * tilde_node_attribute_matrix
        v2 = torch.sum(walk, dim=1)

        walk = torch.bmm(adjacent_matrix, walk) * tilde_node_attribute_matrix
        v3 = torch.sum(walk, dim=1)

        walk = torch.bmm(adjacent_matrix, walk) * tilde_node_attribute_matrix
        v4 = torch.sum(walk, dim=1)

        walk = torch.bmm(adjacent_matrix, walk) * tilde_node_attribute_matrix
        v5 = torch.sum(walk, dim=1)

        walk = torch.bmm(adjacent_matrix, walk) * tilde_node_attribute_matrix
        v6 = torch.sum(walk, dim=1)

        embedded_graph_matrix = torch.stack([v1, v2, v3, v4, v5, v6], dim=1)

        # if torch.cuda.is_available():
        #     tilde_node_attribute_matrix = tilde_node_attribute_matrix.cpu()
        #     embedded_graph_matrix = embedded_graph_matrix.cpu()
        X_embed.extend(tilde_node_attribute_matrix.data.numpy())
        embedded_graph_matrix_list.extend(embedded_graph_matrix.data.numpy())

    embedded_node_matrix_list = np.array(X_embed)
    embedded_graph_matrix_list = np.array(embedded_graph_matrix_list)
    print('embedded_node_matrix_list: ', embedded_node_matrix_list.shape)
    print('embedded_graph_matrix_list shape: {}'.format(embedded_graph_matrix_list.shape))

    return embedded_node_matrix_list, embedded_graph_matrix_list


def graph_embedding(train_path=None,test_path=None,weight_path=None,infer=False,infer_path=None):
    feature_num = 42
    segmentation_list = [range(0, 10), range(10, 17), range(17, 24), range(24, 30), range(30, 36),
                         range(36, 38), range(38, 40), range(40, 42)]
    segmentation_num = len(segmentation_list)


    for embedding_dimension in [50]:
        #创建节点嵌入模型的实例
        model = CBoW(feature_num=feature_num, embedding_dim=embedding_dimension,
                     task_num=segmentation_num, task_size_list=segmentation_list)
        #加载训练参数
        model.load_state_dict(torch.load(weight_path))
        print("节点模型加载成功")
        #CBOW模型参数不变
        model.eval()
        if infer==False:
            for i in ['train','test']:
                if i=='train':
                    data_path=train_path
                else:
                    data_path = test_path
                adjacent_matrix_list, distance_matrix_list, bond_attribute_matrix_list, node_attribute_matrix_list, kwargs = get_data(data_path)
                dataset = GraphDataset(node_attribute_matrix_list=node_attribute_matrix_list, adjacent_matrix_list=adjacent_matrix_list, distance_matrix_list=distance_matrix_list)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

                embedded_node_matrix_list, embedded_graph_matrix_list = get_walk_representation(dataloader,model)
                # print('embedded_graph_matrix_list\t', embedded_graph_matrix_list.shape)
                out_file_path = get_resource_path(f"./tmp/graph_embedding_{i}")

                kwargs['adjacent_matrix_list'] = adjacent_matrix_list
                kwargs['distance_matrix_list'] = distance_matrix_list
                kwargs['embedded_node_matrix_list'] = embedded_node_matrix_list
                kwargs['embedded_graph_matrix_list'] = embedded_graph_matrix_list
                np.savez_compressed(out_file_path, **kwargs)
                print(kwargs.keys())
        else:
            adjacent_matrix_list, distance_matrix_list, bond_attribute_matrix_list, node_attribute_matrix_list, kwargs = get_data(
                infer_path)
            dataset = GraphDataset(node_attribute_matrix_list=node_attribute_matrix_list,
                                   adjacent_matrix_list=adjacent_matrix_list, distance_matrix_list=distance_matrix_list)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
            embedded_node_matrix_list, embedded_graph_matrix_list = get_walk_representation(dataloader, model)
            out_file_path = get_resource_path("./tmp/graph_embedding_infer")
            kwargs['adjacent_matrix_list'] = adjacent_matrix_list
            kwargs['distance_matrix_list'] = distance_matrix_list
            kwargs['embedded_node_matrix_list'] = embedded_node_matrix_list
            kwargs['embedded_graph_matrix_list'] = embedded_graph_matrix_list
            np.savez_compressed(out_file_path, **kwargs)
            print(kwargs.keys())
    n_gram_num=6
    if infer==False:
        X_train = extract_feature_and_label_npy([get_resource_path("./tmp/graph_embedding_train.npz")],
                                                feature_name='embedded_graph_matrix_list',
                                                n_gram_num=n_gram_num)
        X_test = extract_feature_and_label_npy([get_resource_path("./tmp/graph_embedding_test.npz")],
                                               feature_name='embedded_graph_matrix_list',
                                               n_gram_num=n_gram_num)
        with open(get_resource_path("./tmp/X_train_d2.pkl"), "wb") as f:
            pickle.dump(X_train, f)
        with open(get_resource_path("./tmp/X_test_d2.pkl"), "wb") as f:
            pickle.dump(X_test, f)
    else:
        vec_d2 = extract_feature_and_label_npy(
            [get_resource_path(f"./tmp/graph_embedding_infer.npz")],
            feature_name='embedded_graph_matrix_list',
            n_gram_num=n_gram_num)
        return vec_d2
