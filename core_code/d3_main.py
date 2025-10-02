import pandas as pd
from sklearn.model_selection import train_test_split
from pahelix.datasets import InMemoryDataset

from pahelix.utils.data_utils import save_data_list_to_npz
from train import get_resource_path
from core_code.featurizer import DownstreamTransformFn


def load_bioactivity_dataset(data_path):
    """
    data_path：输入要求是excel文件，其中包含两列smiles和label
    Returns:
        an InMemoryDataset instance.
    """
    input_df=pd.read_csv(data_path)
    smiles_list = input_df['smiles']
    labels = input_df['bioactivity_class']
    data_list = []
    for i in range(len(smiles_list)):
        if smiles_list[i] is None:
            continue
        #data是一个字典，它有2个key
        data = {}
        data['smiles'] = smiles_list[i]
        data['bioactivity_class'] = labels.values[i]
        #把字典放入列表中
        data_list.append(data)
    #InMemoryDataset类要求的创建dataset的传入参数是[字典1，字典2...]
    dataset = InMemoryDataset(data_list)
    return dataset

def _save_npz_data(data_list, data_path, max_num_per_file=10000):
    n = len(data_list)
    for i in range(int((n - 1) / max_num_per_file) + 1):
        sub_data_list = data_list[i * max_num_per_file: (i + 1) * max_num_per_file]
        save_data_list_to_npz(sub_data_list, data_path)

def ectrat_d3(data_path,label):
    #把.xlsx的分子文件存入到dataset中
    dataset = load_bioactivity_dataset(data_path)
    #提取特征。提取到的特征放到dataset中一个叫data的列表字典中
    #dict_keys(['atomic_num', 'chiral_tag', 'degree', 'explicit_valence', 'formal_charge', 'hybridization', 'implicit_valence', 'is_aromatic', 'total_numHs', 'mass', 'bond_dir', 'bond_type', 'is_in_ring', 'edges', 'morgan_fp', 'maccs_fp', 'daylight_fg_counts', 'atom_pos', 'bond_length', 'BondAngleGraph_edges', 'bond_angle', 'smiles'])
    dataset.transform(DownstreamTransformFn(), num_workers=1)
    #把smiles+特征存入npz文件中。名字会是part-000000.npz
    dataset.save_data(get_resource_path("./tmp"))

    #分割npz文件
    # InMemoryDataset能通过三种方法创建data列表、npz_data_path、npz_data_file
    #npz_data_path被我改过，写路径直接精确到地址。如果使用files则需要外面加一个列表[]
    dataset = InMemoryDataset(
        npz_data_files=[get_resource_path("./tmp/part-000000.npz")])
    # data是一个列表字典
    data = dataset._load_npz_data_files(["./tmp/part-000000.npz"])

    #和mol2vec对比一下，看看长度是否一致
    print(len(data))

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1, stratify=label, random_state=27)
    #与mol2vec划分的长度对比
    print(len(X_train), len(X_test))
    _save_npz_data(X_train, get_resource_path("./tmpX_train_d3.npz"))
    _save_npz_data(X_test, get_resource_path("./tmpX_test_d3.npz"))

