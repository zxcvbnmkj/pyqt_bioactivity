import os
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles
from sklearn.model_selection import train_test_split
from train_util.little_uitls import num_atom_features, num_bond_features, extract_atom_features, get_atom_distance
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np


def get_2d(data_pd,my_dataset,max_atoms=100,device="cpu",if_infer=False):
    if not if_infer:
        out_file_path = f"./{my_dataset}_Intermediate/X_train_d2feature.npz"
    else:
        out_file_path = f"./{my_dataset}_Intermediate/X_test_d2feature.npz"
    #（1）
    morgan_fps = []
    valid_index = []
    index_list = data_pd.index.tolist()
    smiles_list = data_pd['smiles'].tolist()
    for idx, smiles in zip(index_list, smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if len(mol.GetAtoms()) > max_atoms:
            print('Outlier {} has {} atoms'.format(idx, mol.GetNumAtoms()))
            continue
        # 有效的分子
        valid_index.append(idx)
        fingerprints = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        morgan_fps.append(fingerprints.ToBitString())
    data_pd = data_pd.loc[valid_index]
    data_pd['Fingerprints'] = morgan_fps

    # （2）
    extract_save_info(data_pd,out_file_path,max_atoms)

    # （3）
    # 只有训练集需要
    if not if_infer:
        node_embedding(out_file_path,my_dataset,device)

    #（4）
    if not if_infer:
        graph_embedding(out_file_path,my_dataset,max_atoms,device,if_infer)
    else:
        graph_embedding(out_file_path,my_dataset,max_atoms,device,if_infer)



    #The main function calls this function, so the relative path is the path relative to main.py
    #Delete intermediate files
    files_path = [f"./{my_dataset}_Intermediate/graph_embedding_test.npz",
                  f"./{my_dataset}_Intermediate/graph_embedding_train.npz",
                  f"./{my_dataset}_Intermediate/X_test_d2feature.npz",
                  f"./{my_dataset}_Intermediate/X_train_d2feature.npz"]
    for i in files_path:
        if os.path.exists(i):
            os.remove(i)


def extract_save_info(data_pd,out_file_path, max_atom_num=100,is_infer=False,smiles_list=None):
    import os
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

    if is_infer==False:
        smiles_list = data_pd['smiles'].tolist()


    symbol_candidates = set()
    atom_attribute_dim = num_atom_features()
    # bond_attribute_dim = num_bond_features()

    node_attribute_matrix_list = []
    bond_attribute_matrix_list = []
    adjacent_matrix_list = []
    distance_matrix_list = []
    valid_index = []

    ###
    degree_set = set()
    h_num_set = set()
    implicit_valence_set = set()
    charge_set = set()
    ###

    for line_idx, smiles in enumerate(smiles_list):
        smiles = smiles.strip()
        mol = MolFromSmiles(smiles)
        AllChem.Compute2DCoords(mol)
        conformer = mol.GetConformers()[0]
        feats = factory.GetFeaturesForMol(mol)
        acceptor_atom_ids = map(lambda x: x.GetAtomIds()[0], filter(lambda x: x.GetFamily() =='Acceptor', feats))
        donor_atom_ids = map(lambda x: x.GetAtomIds()[0], filter(lambda x: x.GetFamily() =='Donor', feats))

        adjacent_matrix = np.zeros((max_atom_num, max_atom_num))
        adjacent_matrix = adjacent_matrix.astype(int)
        distance_matrix = np.zeros((max_atom_num, max_atom_num))
        node_attribute_matrix = np.zeros((max_atom_num, atom_attribute_dim))
        node_attribute_matrix = node_attribute_matrix.astype(int)

        if len(mol.GetAtoms()) > max_atom_num:
            print('Outlier {} has {} atoms'.format(line_idx, mol.GetNumAtoms()))
            continue
        valid_index.append(line_idx)

        atom_positions = [None for _ in range(mol.GetNumAtoms()+1)]
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            symbol_candidates.add(atom.GetSymbol())
            atom_positions[atom_idx] = conformer.GetAtomPosition(atom_idx)
            degree_set.add(atom.GetDegree())
            h_num_set.add(atom.GetTotalNumHs())
            implicit_valence_set.add(atom.GetImplicitValence())
            charge_set.add(atom.GetFormalCharge())
            node_attribute_matrix[atom_idx] = extract_atom_features(atom,
                                                                    is_acceptor=atom_idx in acceptor_atom_ids,
                                                                    is_donor=atom_idx in donor_atom_ids)
        node_attribute_matrix_list.append(node_attribute_matrix)

        for idx_i in range(mol.GetNumAtoms()):
            for idx_j in range(idx_i+1, mol.GetNumAtoms()):
                distance = get_atom_distance(conformer.GetAtomPosition(idx_i),
                                             conformer.GetAtomPosition(idx_j))
                distance_matrix[idx_i, idx_j] = distance
                distance_matrix[idx_j, idx_i] = distance
        distance_matrix_list.append(distance_matrix)

        for bond in mol.GetBonds():
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            begin_index = begin_atom.GetIdx()
            end_index = end_atom.GetIdx()
            adjacent_matrix[begin_index, end_index] = 1
            adjacent_matrix[end_index, begin_index] = 1
        adjacent_matrix_list.append(adjacent_matrix)

    adjacent_matrix_list = np.asarray(adjacent_matrix_list)
    distance_matrix_list = np.asarray(distance_matrix_list)
    node_attribute_matrix_list = np.asarray(node_attribute_matrix_list)
    bond_attribute_matrix_list = np.asarray(bond_attribute_matrix_list)
    np.savez_compressed(out_file_path,
                        adjacent_matrix_list=adjacent_matrix_list,
                        distance_matrix_list=distance_matrix_list,
                        node_attribute_matrix_list=node_attribute_matrix_list,
                        bond_attribute_matrix_list=bond_attribute_matrix_list)



def node_embedding(train_path,my_dataset,device):
    epochs = 20
    weight_file = f"./{my_dataset}_Intermediate/CBoW_50dim.pt"
    random_dimension = 50
    segmentation_num = 8
    padding_size = 10

    segmentation_list = [range(0, 10), range(10, 17), range(17, 24), range(24, 30), range(30, 36),
                         range(36, 38), range(38, 40), range(40, 42)]
    model = CBoW(feature_num=42, embedding_dim=random_dimension,
                 task_num=segmentation_num, task_size_list=segmentation_list)
    #whether or not use cuda（1）
    if device=="cuda":
        model.cuda()
        print("The CUDA device currently in use is device ",{torch.cuda.current_device()})
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    train_dataset = GraphDataset(train_path,padding_size=padding_size)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    train(model,train_dataloader,optimizer,epochs,weight_file,device)

#CBoW model
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

class GraphDataset(Dataset):
    def __init__(self,data_path,padding_size):
        self.X_data, self.Y_label_list = [], []

        X_data, Y_label_list = get_GraphDataset_data(data_path=data_path, padding_size=padding_size)
        self.X_data.extend(X_data)
        self.Y_label_list.extend(Y_label_list)
        self.X_data = np.array(self.X_data)
        self.Y_label_list = np.array(self.Y_label_list)


    def __len__(self):

        return len(self.X_data)

    def __getitem__(self, idx):
        x_data = self.X_data[idx]
        y_label_list = self.Y_label_list[idx]

        x_data = torch.from_numpy(x_data)
        y_label_list = torch.from_numpy(y_label_list)
        return x_data, y_label_list

def get_GraphDataset_data(data_path, padding_size,max_atom_num=100):
    FEATURE_NUM=42
    segmentation_list = [range(0, 10), range(10, 17), range(17, 24), range(24, 30), range(30, 36),
                         range(36, 38), range(38, 40), range(40, 42)]
    data = np.load(data_path)
    adjacent_matrix_list = data['adjacent_matrix_list']
    node_attribute_matrix_list = data['node_attribute_matrix_list']

    molecule_num = adjacent_matrix_list.shape[0]

    X_data = []
    Y_label_list = []
    for adjacent_matrix, node_attribute_matrix in zip(adjacent_matrix_list, node_attribute_matrix_list):
        assert len(adjacent_matrix) == max_atom_num
        assert len(node_attribute_matrix) == max_atom_num
        for i in range(max_atom_num):
            if sum(adjacent_matrix[i]) == 0:
                break
            x_temp = np.zeros((padding_size, FEATURE_NUM))
            cnt = 0
            for j in range(max_atom_num):
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

def train(model,train_dataloader,optimizer,epochs,weight_file,device):
    criterion = nn.CrossEntropyLoss()
    model.train()

    optimal_loss = 1e7
    for epoch in range(epochs):
        train_loss = []

        for batch_id, (x_data, y_actual) in enumerate(train_dataloader):
            x_data = Variable(x_data).float()
            y_actual = Variable(y_actual).long()


            #whether or not use cuda（2）
            if device=="cuda":
                x_data = x_data.cuda()
                y_actual = y_actual.cuda()



            optimizer.zero_grad()
            y_predict = model(x_data)

            loss = 0
            for i in range(8):
                y_true, y_pred = y_actual[..., i], y_predict[i]
                temp_loss = criterion(y_pred, y_true)
                loss += temp_loss
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        train_loss = np.mean(train_loss)
        if train_loss < optimal_loss:
            optimal_loss = train_loss
            torch.save(model.state_dict(), weight_file)
    return


def test(dataloader,model,device):
    model.eval()
    accuracy, total = 0, 0
    for batch_id, (x_data, y_actual) in enumerate(dataloader):
        x_data = Variable(x_data).float()
        y_actual = Variable(y_actual).long()

        #whether or not use cuda（3）
        if device=="cuda":
            x_data = x_data.cuda()
            y_actual = y_actual.cuda()


        y_predict = model(x_data)

        for i in range(8):
            y_true, y_pred = y_actual[..., i].cpu().data.numpy(), y_predict[i].cpu().data.numpy()
            y_pred = y_pred.argmax(1)
            accuracy += np.sum(y_true == y_pred)
            total += y_pred.shape[0]
    return


#（4）边嵌入
def graph_embedding(data_path,my_dataset,max_atoms,device,if_infer):
    data_type="test" if if_infer else "train"
    out_file_path=f"./{my_dataset}_Intermediate/graph_embedding_{data_type}"
    embedding_dimension_list = [50]

    feature_num = 42
    # max_atom_num = max_atoms
    segmentation_list = [range(0, 10), range(10, 17), range(17, 24), range(24, 30), range(30, 36),
                         range(36, 38), range(38, 40), range(40, 42)]
    segmentation_num = len(segmentation_list)


    for embedding_dimension in embedding_dimension_list:
        # 加载预训练节点嵌入模型
        model = CBoW(feature_num=feature_num, embedding_dim=embedding_dimension,
                     task_num=segmentation_num, task_size_list=segmentation_list)
        weight_file = f"./{my_dataset}_Intermediate/CBoW_50dim.pt"
        model.load_state_dict(torch.load(weight_file))
        #whether or not use cuda（4）
        if device=="cuda":
            model.cuda()

        model.eval()
        # 为训练集或者测试集生成图嵌入
        adjacent_matrix_list, distance_matrix_list, bond_attribute_matrix_list, node_attribute_matrix_list, kwargs = get_data(data_path)
        dataset = Graph_embed_Dataset(node_attribute_matrix_list=node_attribute_matrix_list, adjacent_matrix_list=adjacent_matrix_list, distance_matrix_list=distance_matrix_list)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

        embedded_node_matrix_list, embedded_graph_matrix_list = get_walk_representation(dataloader,model,device)
        kwargs['adjacent_matrix_list'] = adjacent_matrix_list
        kwargs['distance_matrix_list'] = distance_matrix_list
        kwargs['embedded_node_matrix_list'] = embedded_node_matrix_list
        kwargs['embedded_graph_matrix_list'] = embedded_graph_matrix_list
        np.savez_compressed(out_file_path, **kwargs)
    n_gram_num=6

    X = extract_feature_and_label_npy([f"{out_file_path}.npz"],
                                            feature_name='embedded_graph_matrix_list',
                                            n_gram_num=n_gram_num)
    if not if_infer:
        with open(f"./{my_dataset}_Intermediate/X_train_d2.pkl", "wb") as f:
            pickle.dump(X, f)
    else:
        with open(f"./{my_dataset}_Intermediate/X_test_d2.pkl", "wb") as f:
            pickle.dump(X, f)


class Graph_embed_Dataset(Dataset):
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


def get_data(data_path):
    data = np.load(data_path)
    adjacent_matrix_list = data['adjacent_matrix_list']
    distance_matrix_list = data['distance_matrix_list']
    bond_attribute_matrix_list = data['bond_attribute_matrix_list']
    node_attribute_matrix_list = data['node_attribute_matrix_list']

    kwargs = {}

    return adjacent_matrix_list, distance_matrix_list, bond_attribute_matrix_list,\
           node_attribute_matrix_list, kwargs

def extract_feature_and_label_npy(data_file_list, feature_name, n_gram_num):
    X_data = []
    for data_file in data_file_list:
        data = np.load(data_file)
        X_data_temp = data[feature_name]
        if X_data_temp.ndim == 4:
            X_data_temp = X_data_temp[:, :n_gram_num, ...]

            molecule_num, _, embedding_dimension, segmentation_num = X_data_temp.shape

            X_data_temp = X_data_temp.reshape((molecule_num, n_gram_num*embedding_dimension*segmentation_num), order='F')

        elif X_data_temp.ndim == 3:
            X_data_temp = X_data_temp[:, :n_gram_num, ...]
            molecule_num, _, embedding_dimension = X_data_temp.shape
            X_data_temp = X_data_temp.reshape((molecule_num, n_gram_num*embedding_dimension), order='F')


        X_data.extend(X_data_temp)
    X_data = np.stack(X_data)
    return X_data

def get_walk_representation(dataloader,model,device):
    X_embed = []
    embedded_graph_matrix_list = []
    for batch_id, (node_attribute_matrix, adjacent_matrix, distance_matrix) in enumerate(dataloader):
        node_attribute_matrix = Variable(node_attribute_matrix).float()
        adjacent_matrix = Variable(adjacent_matrix).float()

        # whether or not use cuda（5）
        if device=="cuda":
            node_attribute_matrix = node_attribute_matrix.cuda()
            adjacent_matrix = adjacent_matrix.cuda()

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

        # whether or not use cuda (6)
        if device=="cuda":
            #Move the tensor in cuda to the CPU in order to convert it to NumPy type
            tilde_node_attribute_matrix = tilde_node_attribute_matrix.cpu()
            embedded_graph_matrix = embedded_graph_matrix.cpu()

        X_embed.extend(tilde_node_attribute_matrix.data.numpy())
        embedded_graph_matrix_list.extend(embedded_graph_matrix.data.numpy())

    embedded_node_matrix_list = np.array(X_embed)
    embedded_graph_matrix_list = np.array(embedded_graph_matrix_list)
    return embedded_node_matrix_list, embedded_graph_matrix_list