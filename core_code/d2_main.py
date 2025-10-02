import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles
from sklearn.model_selection import train_test_split
import os
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures

# 对本项目文件的依赖
from core_code.d2_utils.tools import get_atom_distance, extract_atom_features, num_atom_features, get_resource_path
from core_code.d2_utils.graph_embedding import graph_embedding
from core_code.d2_utils.node_embedding import node_embedding


# y_label要求df类型
def get_d2(data_pd,y_label):
    max_atoms=100
    # (2) 第一步
    #转换成功的分子对应的ECFP
    morgan_fps = []
    # 转换成功的原子编号
    valid_index = []
    #这一步是想提取出摩根指纹与其对应的分子编号
    index_list = data_pd.index.tolist()
    smiles_list = data_pd['smiles'].tolist()
    # max_atoms=-1
    for idx, smiles in zip(index_list, smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        # mol_length=len(mol.GetAtoms())
        # if mol_length > max_atoms:
        #     max_atoms=mol_length
        valid_index.append(idx)
        fingerprints = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        morgan_fps.append(fingerprints.ToBitString())
    data_pd = data_pd.loc[valid_index]
    data_pd['Fingerprints'] = morgan_fps
    print(f"数据集中最大的原子数是{max_atoms}")
    X_train, X_test, y_train, y_test = train_test_split(data_pd, y_label, test_size=0.1, stratify=y_label, random_state=27)

    # (2) 第二步
    # 从分子的2D结构中提取图神经网络所需的图结构数据
    # 输入: SMILES分子数据
    # 输出: 图神经网络需要的四个矩阵（邻接矩阵、距离矩阵、节点属性矩阵、边属性矩阵），保存为npz文件
    extract_graph_train(X_train,get_resource_path("./tmp/X_train_d2feature.npz"),max_atom_num=max_atoms)
    extract_graph_train(X_test, get_resource_path("./tmp/X_test_d2feature.npz"),max_atom_num=max_atoms)

    # (3) 第三步：节点嵌入
    node_embedding(get_resource_path("./tmp/X_train_d2feature.npz"),
                   get_resource_path("./tmp/X_test_d2feature.npz"),
                   weight_path=get_resource_path("./tmp/CBoW_50dim.pt"), max_atoms=max_atoms)
    # （4） 第四步：图嵌入
    graph_embedding(get_resource_path("./tmp/X_train_d2feature.npz"),
                    get_resource_path("./tmp/X_test_d2feature.npz"),
                    weight_path=get_resource_path("./tmp/CBoW_50dim.pt"))
    # (5) 删除中间文件
    # Cbow 模型不能删除
    files_path = [get_resource_path("./tmp/graph_embedding_test.npz"),
                  get_resource_path("./tmp/graph_embedding_train.npz"),
                  get_resource_path("./tmp/X_train_d2feature.npz"),
                  get_resource_path("./tmp/X_test_d2feature.npz")]
    for i in files_path:
        if os.path.exists(i):
            os.remove(i)

def get_d2_batch_infer(data_pd):
    max_atoms = 100
    # (2) 第一步
    #转换成功的分子对应的ECFP
    # morgan_fps = []
    # # 转换成功的原子编号
    # valid_index = []
    #这一步是想提取出摩根指纹与其对应的分子编号
    # index_list = data_pd.index.tolist()
    smiles_list = data_pd['smiles'].tolist()
    # max_atoms=-1
    # for idx, smiles in zip(index_list, smiles_list):
    #     mol = Chem.MolFromSmiles(smiles)
    #     mol_length=len(mol.GetAtoms())
    #     if mol_length > max_atoms:
    #         max_atoms=mol_length
    #     valid_index.append(idx)
    #     fingerprints = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    #     morgan_fps.append(fingerprints.ToBitString())
    # data_pd = data_pd.loc[valid_index]
    # data_pd['Fingerprints'] = morgan_fps
    # print(f"数据集中最大的原子数是{max_atoms}")

    # (2) 第二步
    # 从分子的2D结构中提取图神经网络所需的图结构数据
    # 输入: SMILES分子数据
    # 输出: 图神经网络需要的四个矩阵（邻接矩阵、距离矩阵、节点属性矩阵、边属性矩阵），保存为npz文件
    extract_graph_train(None, get_resource_path("./tmp/batch_infer_d2feature.npz"),max_atom_num=max_atoms,is_infer=True,smiles_list=smiles_list)

    # （4） 第四步：图嵌入
    vec2 = graph_embedding(weight_path=get_resource_path("./tmp/CBoW_50dim.pt"), infer=True,
                           infer_path=get_resource_path("./tmp/batch_infer_d2feature.npz"))

    # (5) 删除中间文件
    files_path = [get_resource_path("./tmp/batch_infer_d2feature.npz")]
    for i in files_path:
        if os.path.exists(i):
            os.remove(i)
    return vec2

def extract_graph_train(data_pd=None,out_file_path=None, max_atom_num=500,is_infer=False,smiles_list=None):
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

    if is_infer==False:
        smiles_list = data_pd['smiles'].tolist()

    symbol_candidates = set()
    atom_attribute_dim = num_atom_features()

    node_attribute_matrix_list = []
    bond_attribute_matrix_list = []
    adjacent_matrix_list = []
    distance_matrix_list = []
    valid_index = []

    degree_set = set()
    h_num_set = set()
    implicit_valence_set = set()
    charge_set = set()

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
    #相邻矩阵(408, 132, 132)
    print('adjacent matrix shape\t', adjacent_matrix_list.shape)
    #距离矩阵(408, 132, 132)
    print('distance matrix shape\t', distance_matrix_list.shape)
    #节点属性(408, 132, 42)
    print('node attr matrix shape\t', node_attribute_matrix_list.shape)
    #边属性(0,)
    print('bond attr matrix shape\t', bond_attribute_matrix_list.shape)
    #{'Na', 'P', 'C', 'H', 'S', 'O', 'N', 'Cl', 'Br', 'F'}
    print(symbol_candidates)
    #验证集中几个有效
    #408 valid out of 408，全有效
    print('{} valid out of {}'.format(len(valid_index), len(smiles_list)))
    #degree set:	 {0, 1, 2, 3, 4}。5折
    print('degree set:\t', degree_set)
    #h num set: 	 {0, 1, 2, 3}
    print('h num set: \t', h_num_set)
    #含蓄价值{0, 1, 2, 3}
    print('implicit valence set: \t', implicit_valence_set)
    # {0, 1, -1}
    print('charge set:\t', charge_set)

    np.savez_compressed(out_file_path,
                        adjacent_matrix_list=adjacent_matrix_list,
                        distance_matrix_list=distance_matrix_list,
                        node_attribute_matrix_list=node_attribute_matrix_list,
                        bond_attribute_matrix_list=bond_attribute_matrix_list)
    print("图结构构建所需的数据提取完毕")