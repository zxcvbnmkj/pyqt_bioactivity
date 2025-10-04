import pandas as pd
from pahelix.datasets import InMemoryDataset
from pahelix.utils.data_utils import save_data_list_to_npz
def _save_npz_data(data_list, data_path, max_num_per_file=10000):
    n = len(data_list)
    for i in range(int((n - 1) / max_num_per_file) + 1):
        sub_data_list = data_list[i * max_num_per_file: (i + 1) * max_num_per_file]
        save_data_list_to_npz(sub_data_list, data_path)

def load_bioactivity_dataset(data_path):
    # Data_path requires the path to an Excel file, which includes two columns of files and label
    infer=True
    input_df=pd.read_excel(data_path)
    smiles_list = input_df['smiles']
    if 'label' in input_df.columns:
        labels = input_df['label']
        infer = False
    data_list = []
    for i in range(len(smiles_list)):
        if smiles_list[i] is None:
            continue
        data = {}
        data['smiles'] = smiles_list[i]
        if not infer:
            data['label'] = labels.values[i]
        data_list.append(data)
    dataset = InMemoryDataset(data_list)
    return dataset
