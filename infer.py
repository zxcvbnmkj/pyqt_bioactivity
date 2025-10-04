import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import paddle
from pahelix.model_zoo.gem_model import GeoGNNModel
from pahelix.utils import load_json_config
from train import extract_feat
from train_util.featurizer import DownstreamCollateFn
import pickle
from pahelix.datasets import InMemoryDataset
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import Dataset

class Infer_Dataset(Dataset):
    def __init__(self,d1_file,gram_data_file):
        with open(d1_file, "rb") as f:
            self.molvec= pickle.load(f)
        with open(gram_data_file, "rb") as f:
            self.gram = pickle.load(f)


    def __getitem__(self, index):
        data1=self.gram[index]
        #need  tensor type
        data1=torch.Tensor(data1)
        data1 = data1.unsqueeze(0)

        data2=self.molvec[index]
        data2 = torch.Tensor(data2)
        data2 = data2.unsqueeze(0)

        return data1,data2,torch.LongTensor([index])

    def __len__(self):
        return len(self.molvec)


def infer_model_forward(model,input0,input1,input2,device):
    input2 = input2.tolist()
    input2 = torch.tensor(input2)
    if device=="cuda":
        input0=input0.cuda()
        input1=input1.cuda()
        input2=input2.cuda()
    fusion_out, d1_logits, d3_logits,gram_logits, d1_conf, d3_conf,gram_conf=model(input0,input1,input2)
    return fusion_out



def model_test(model,test_dataloader,geo_loader_test,compound_encoder,norm,device):
    epoch_test_y_predict = []
    epoch_test_y_prob=[]
    predict_value=[]
    model.eval()
    softmax = torch.nn.Softmax()
    with torch.no_grad():
        for data, geo_data in zip(test_dataloader, geo_loader_test):
            gram, mol_vec, _ = data
            atom_bond_graphs, bond_angle_graphs = geo_data
            node_repr, edge_repr, graph_repr = compound_encoder(atom_bond_graphs.tensor(), bond_angle_graphs.tensor())
            graph_repr = norm(graph_repr)
            outputs= infer_model_forward(model, gram, mol_vec, graph_repr,device)

            if device=="cuda":
                outputs=outputs.cpu().detach()
            outputs = softmax(outputs)
            y_predict = outputs.argmax(-1)
            epoch_test_y_predict.extend(y_predict)
            epoch_test_y_prob.extend(outputs)
    # 打印分子活性预测值
    prob_value = [tensor_value for tensor_value in epoch_test_y_prob]
    # predict_value = [tensor_value.item() for tensor_value in epoch_test_y_predict]
    predict_value.extend(["非活性" if tensor_value.item() == 0 else "活性" for tensor_value in epoch_test_y_predict])
    print(prob_value)
    print(predict_value)
    return [prob_value,predict_value]
    # df=pd.read_excel("./测试集.xlsx")
    # df["预测"] = ["活性" if value == 1 else "非活性" for value in predict_value]
    # df.to_excel("./预测结果.xlsx",index=False)


def infer(batch_size=32,my_dataset="tmp",device="cpu",save_model_path=None):
    torch.manual_seed(42)
    paddle.seed(42)
    np.random.seed(42)

    compound_encoder_config = load_json_config("./train_util/gnnconfig.json")
    compound_encoder = GeoGNNModel(compound_encoder_config)
    compound_encoder.set_state_dict(paddle.load("./train_util/class.pdparams"))
    # 请将X_test_d3.npz替换成需要预测的分子文件的三维信息文件名字
    test_dataset = InMemoryDataset(
        npz_data_files=[f"./{my_dataset}_Intermediate/X_test_d3.npz"])
    collate_fn = DownstreamCollateFn(
        atom_names=compound_encoder_config['atom_names'],
        bond_names=compound_encoder_config['bond_names'],
        bond_float_names=compound_encoder_config['bond_float_names'],
        bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
        task_type='class', is_inference=True)
    geo_loader_test = test_dataset.get_data_loader(
        batch_size=batch_size,
        # 强制单线程
        num_workers=1,
        collate_fn=collate_fn)
    model = torch.load(f"{save_model_path}",weights_only=False)
    if device=="cuda":
        model.cuda()
    # 请将X_test_d3.npz替换成需要预测的分子文件的一维、二维信息文件名字
    test_set = Infer_Dataset(f"./{my_dataset}_Intermediate/X_test_d1.pkl",
                         f"./{my_dataset}_Intermediate/X_test_d2.pkl")
    test_dataloader = DataLoader(test_set, batch_size=batch_size)

    norm = paddle.nn.LayerNorm(compound_encoder.graph_dim)

    result=model_test(model, test_dataloader=test_dataloader, geo_loader_test=geo_loader_test,
         compound_encoder=compound_encoder,
         norm=norm,device=device)
    # files_path = [f"./{my_dataset}_Intermediate/X_test_d1.pkl",
    #               f"./{my_dataset}_Intermediate/X_test_d2.pkl",
    #               f"./{my_dataset}_Intermediate/X_test_d3.npz"]
    # for i in files_path:
    #     if os.path.exists(i):
    #         os.remove(i)
    # return result

def batch_predict_begin(my_dataset="tmp",device="cpu",compounds_files=None,batch_size=32,save_model_path=None):
    # extract_feat(my_dataset=my_dataset,if_infer=True,device=device,compounds_files=compounds_files)
    result=infer(batch_size=batch_size,my_dataset=my_dataset,device=device,save_model_path=save_model_path)
    return result

def predict_single(smiles,my_dataset="tmp",device="cpu",save_model_path=None):
    df = pd.DataFrame([smiles], columns=['smiles'])
    save_single=f"./{my_dataset}_Intermediate/single.xlsx"
    df.to_excel(save_single,index=False)
    extract_feat(my_dataset=my_dataset,if_infer=True,device=device,compounds_files=save_single)
    result=infer(my_dataset=my_dataset,device=device,batch_size=32,save_model_path=save_model_path)
    print(result)
    print("分子预测完毕！\n它为非活性的概率是{:.5f}，为活性的概率是{:.5f}\n最终的预测结果是{}".format(
            result[0][0][0], result[0][0][1],result[1][0]))
    return result

if __name__ == '__main__':
    # 测试批量处理
    # batch_predict_begin(compounds_files=r"E:\pyqt_bio_v3.0\test_data\2025-2-28_newsun\test.xlsx",save_model_path=r"E:\pyqt_bio_v3.0\tmp_Intermediate\best_model.pth")
    # batch_predict_begin(compounds_files=r"E:\pyqt_bio_v3.0\test_data\2025-2-28_newsun\test.xlsx",save_model_path=r"E:\pyqt_bio_v3.0\tmp_Intermediate\best_model.pth")

    batch_predict_begin(my_dataset="tmp",compounds_files=r"E:\pyqt_bio_v3.0\test_data\2025-2-28_newsun\test.xlsx",save_model_path=r"E:\pyqt_bio_v3.0\tmp2_Intermediate\best_model.pth")
    batch_predict_begin(my_dataset="tmp",compounds_files=r"E:\pyqt_bio_v3.0\test_data\2025-2-28_newsun\test.xlsx",save_model_path=r"E:\pyqt_bio_v3.0\tmp2_Intermediate\best_model.pth")

    # batch_predict_begin(my_dataset="tmp",compounds_files=r"E:\pyqt_bio_v3.0\test_data\2025-2-28_newsun\test.xlsx",save_model_path=r"C:/Users/张丽/Desktop/MIDF-DMAP/AAA_Intermediate/save_model/【0.99425-bs32】best_auc_model.pth")
    # batch_predict_begin(my_dataset="tmp",compounds_files=r"E:\pyqt_bio_v3.0\test_data\2025-2-28_newsun\test.xlsx",save_model_path=r"C:/Users/张丽/Desktop/MIDF-DMAP/AAA_Intermediate/save_model/【0.99425-bs32】best_auc_model.pth")
    # 测试单个
    # predict_single("CC(COS(=O)OCCF)OC1=CC=C(Cl)C=C1Cl",save_model_path=r"E:\pyqt_bio_v3.0\tmp_Intermediate\best_model.pth")
    # predict_single("CC(COS(=O)OCCF)OC1=CC=C(Cl)C=C1Cl",save_model_path=r"E:\pyqt_bio_v3.0\tmp_Intermediate\best_model.pth")

