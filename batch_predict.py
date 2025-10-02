import os
import torch
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, MolSentence, sentences2vec
from rdkit import Chem
from torch.utils.data import DataLoader
from core_code.d2_main import get_d2_batch_infer
from core_code.d2_utils.tools import get_resource_path
from core_code.dataset import batch_infer_Dataset


def batch_predict_begin(ui,df,model,if_3d):
    if ui is not None:
        infer_msg = "批量预测开始！\n"
        ui.textBrowser_infer_info.setText(infer_msg)
    backup_df = df
    # 一维特征提取
    df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    df['ecfp'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)
    wv_model = word2vec.Word2Vec.load(get_resource_path('./Dependencies/model_300dim.pkl'))
    vec1 = []
    for x in sentences2vec(df['ecfp'], wv_model, unseen='UNK'):
        vec1.append(x)
    # 二维特征提取
    if ui is not None:
        infer_msg = infer_msg + "一维特征提取完毕。\n正在提取分子二维层面的特征......\n"
        ui.textBrowser_infer_info.setText(infer_msg)
    vec2 = get_d2_batch_infer(backup_df)
    # 数据集准备
    batch_infer_set = batch_infer_Dataset(vec1, vec2)
    batch_infer_dataloader = DataLoader(batch_infer_set, batch_size=16)
    # 加载模型
    if ui is not None:
        model = torch.load(model)
    model.eval()
    # predict_proba_infer=[]
    predict_all = []
    softmax = torch.nn.Softmax()
    with torch.no_grad():
        for data in batch_infer_dataloader:
            mol_vec,garm = data
            outputs = model(mol_vec,garm)
            outputs = softmax(outputs)
            # predict_proba_infer.extend(outputs[:, 1].numpy())
            pred = outputs.argmax(-1)
            print(outputs)
            print(pred)
            predict_all.extend(["非活性" if i == 0 else "活性" for i in pred])
    # Todo:
    if if_3d:
        pass
    if ui is not None:
        return predict_all
    else:
        print(predict_all)
    files_path = [get_resource_path("./tmp/graph_embedding_infer.npz"),
                 get_resource_path("./tmp/infer_d2feature.npz")]
    for i in files_path:
        if os.path.exists(i):
            os.remove(i)