import pandas as pd
from pahelix.datasets import InMemoryDataset
from pahelix.utils.data_utils import save_data_list_to_npz
from core_code.featurizer import DownstreamTransformFn


def load_bioactivity_dataset(data_path):
    """
    data_path：输入要求是excel文件，其中包含两列smiles和label
    Returns:
        an InMemoryDataset instance.
    """
    input_df=pd.read_excel(data_path)
    smiles_list = input_df['smiles']
    labels = input_df['label']
    data_list = []
    for i in range(len(smiles_list)):
        if smiles_list[i] is None:
            continue
        #data是一个字典，它有2个key
        data = {}
        data['smiles'] = smiles_list[i]
        data['label'] = labels.values[i]
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

def ectrat_d3_infer(smiles):
    data_list=[]
    data = {}
    data['smiles'] =smiles
    data['label'] =0
    # 把字典放入列表中
    data_list.append(data)
    dataset = InMemoryDataset(data_list)

    dataset.transform(DownstreamTransformFn(), num_workers=1)
    #把smiles+特征存入npz文件中。名字会是part-000000.npz
    dataset.save_data("D:\Pythonnnn\pyqt_bioactivity\casp\infer_intermediate")
    vec3=dataset._load_npz_data_files(["D:\Pythonnnn\pyqt_bioactivity\casp\infer_intermediate\part-000000.npz"])
    return vec3

if __name__ == '__main__':
    ectrat_d3_infer("O=C1C=CC(=O)C=C1")
