from __future__ import print_function
import math
import numpy as np
from rdkit import Chem
import os
import sys
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_resource_path(relative_path):
    """获取资源的绝对路径，兼容开发环境和打包后的exe"""
    try:
        # 打包后的exe运行时，sys._MEIPASS会被设置
        base_path = sys._MEIPASS
    except AttributeError:
        # 开发环境，使用当前文件所在目录
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


"""
These are some tool functions for extracting the two-dimensional structure of molecules. This code is based on Python 2 and needs to be partially changed to Python 3 format
This file uses code from the n_gram_graph library (https://github.com/chao1224/n_gram_graph) by chao1224 (https://github.com/chao1224).
Licensed under the MIT License (https://github.com/chao1224/n_gram_graph/blob/master/LICENSE).
"""





atom_candidates = ['C', 'Cl', 'I', 'F', 'O', 'N', 'P', 'S', 'Br', 'Unknown']

possible_hybridization_list = [
    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
]


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: 1 if x == s else 0, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        logging.info('Unknown detected: {}'.format(x))
        x = allowable_set[-1]
    return list(map(lambda s: 1 if x == s else 0, allowable_set))


def extract_atom_features(atom, explicit_H=False, is_acceptor=0, is_donor=0):
    if explicit_H:
        return np.array(one_of_k_encoding_unk(atom.GetSymbol(), atom_candidates) +
                        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) +
                        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) +
                        one_of_k_encoding(atom.GetFormalCharge(), [-2, -1, 0, 1, 2, 3]) +
                        one_of_k_encoding(atom.GetIsAromatic(), [0, 1])
                        )
    else:
        # The map function in Python 2 returns a list type, while in Python 3 it returns a map type. So we need to manually convert it to a list
        return np.array(one_of_k_encoding_unk(atom.GetSymbol(), atom_candidates) +
                        one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) +
                        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6]) +
                        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                        one_of_k_encoding_unk(atom.GetFormalCharge(), [-2, -1, 0, 1, 2, 3]) +
                        one_of_k_encoding(atom.GetIsAromatic(), [0, 1]) +
                        one_of_k_encoding(is_acceptor, [0, 1]) +
                        one_of_k_encoding(is_donor, [0, 1])
                        )


def extract_bond_features(bond):
    bt = bond.GetBondType()
    bond_features = np.array([bt == Chem.rdchem.BondType.SINGLE,
                              bt == Chem.rdchem.BondType.DOUBLE,
                              bt == Chem.rdchem.BondType.TRIPLE,
                              bt == Chem.rdchem.BondType.AROMATIC,
                              bond.GetIsConjugated(),
                              bond.IsInRing()])
    bond_features = bond_features.astype(int)
    return bond_features


def num_atom_features(explicit_H=False):
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(extract_atom_features(a, explicit_H))


def num_bond_features():
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(extract_bond_features(simple_mol.GetBonds()[0]))


def get_atom_distance(atom_a, atom_b):
    return math.sqrt((atom_a.x - atom_b.x) ** 2 + (atom_a.y - atom_b.y) ** 2 + (atom_a.z - atom_b.z) ** 2)


def extract_feature_and_label_npy(data_file_list, feature_name, n_gram_num):
    X_data = []
    for data_file in data_file_list:
        data = np.load(data_file)
        X_data_temp = data[feature_name]
        if X_data_temp.ndim == 4:
            print('original size\t', X_data_temp.shape)
            X_data_temp = X_data_temp[:, :n_gram_num, ...]
            print('truncated size\t', X_data_temp.shape)

            molecule_num, _, embedding_dimension, segmentation_num = X_data_temp.shape

            X_data_temp = X_data_temp.reshape((molecule_num, n_gram_num*embedding_dimension*segmentation_num), order='F')

        elif X_data_temp.ndim == 3:
            print('original size\t', X_data_temp.shape)
            X_data_temp = X_data_temp[:, :n_gram_num, ...]
            print('truncated size\t', X_data_temp.shape)
            molecule_num, _, embedding_dimension = X_data_temp.shape
            X_data_temp = X_data_temp.reshape((molecule_num, n_gram_num*embedding_dimension), order='F')


        X_data.extend(X_data_temp)
    X_data = np.stack(X_data)
    print(X_data.shape)
    print('X data\t', X_data.shape)
    return X_data


def reshape_data_into_2_dim(data):
    if data.ndim == 1:
        n = data.shape[0]
        data = data.reshape(n, 1)
    return data