import sys
import logging
import os
import pickle
import numpy as np
import pandas as pd
import torch
import paddle
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

from batch_predict import batch_predict_begin
from core_code.d2_utils.tools import get_resource_path
from pahelix.datasets.inmemory_dataset import InMemoryDataset
from pyqt5_plugins.examplebuttonplugin import QtGui
from pahelix.model_zoo.gem_model import GeoGNNModel

# 对本项目文件的依赖
from core_code.Focal_Loss import focal_loss
from core_code.MYmodel_final import theModel_final, Model_d2d3
from core_code.dataset import MYDataset
from core_code.featurizer import DownstreamCollateFn

def abnorm(df):
    idx_failed_ecfp = []
    for idx, row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol is None:
                # An exception occurred while converting smiles to mol
                # When RDKit encounters non-compliant smiles, it cannot convert them to mol form and therefore returns None
                idx_failed_ecfp.append(idx)
            else:
                # Try to see if there is an error when converting to ECFP, and if so, throw an exception
                _ = AllChem.GetMorganFingerprint(mol, radius=1)
        except:
            idx_failed_ecfp.append(idx)

    print(f"List of abnormal molecular indexes：{idx_failed_ecfp}")
    # Remove these molecules
    df = df.drop(df.index[idx_failed_ecfp])
    df.reset_index(inplace=True, drop=True, names="num")
    return df

def create_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(
        filename=get_resource_path("./log/1.log"))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger




def train_begin(ui, epoch=60, batch_size=32, lr=0.004, if_3d=False,save_model_path=None):
    BATCH_SIZE = batch_size
    EPOCH = epoch
    RATE = lr
    PATIENCE=30
    OPTIM = 'SGD'
    LOSS_FUN = "focal"

    # 创建日志
    logger = create_logger()
    # 3d的准备部分
    if if_3d:
        compound_encoder_path = get_resource_path("./Dependencies/class.pdparams")
        compound_encoder_config = get_resource_path("./Dependencies/geognn2.json")
        ### build model。这里是在创建类的对象
        compound_encoder = GeoGNNModel(compound_encoder_config)
        # 加载预训练模型
        compound_encoder.set_state_dict(paddle.load(compound_encoder_path))
        train_dataset = InMemoryDataset(
            npz_data_path=get_resource_path("./Dependencies/X_train_d3.npz"))
        test_dataset = InMemoryDataset(
            npz_data_path=get_resource_path("./Dependencies/X_test_d3.npz"))
        collate_fn = DownstreamCollateFn(
            atom_names=compound_encoder_config['atom_names'],
            bond_names=compound_encoder_config['bond_names'],
            bond_float_names=compound_encoder_config['bond_float_names'],
            bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
            task_type='class', is_inference=True)
        geo_loader_train = train_dataset.get_data_loader(
            batch_size=BATCH_SIZE,
            num_workers=1,
            shuffle=False,
            collate_fn=collate_fn)
        geo_loader_test = test_dataset.get_data_loader(
            batch_size=BATCH_SIZE,
            num_workers=1,
            shuffle=False,
            collate_fn=collate_fn)
        norm = paddle.nn.LayerNorm(compound_encoder.graph_dim)
        model = theModel_final()
    else:
        model = Model_d2d3()
    train_set = MYDataset(get_resource_path("./tmp/X_train_d1.pkl"),
                          get_resource_path("./tmp/X_train_d2.pkl"),
                          get_resource_path("./tmp/y_train.pkl"))

    test_set = MYDataset(get_resource_path("./tmp/X_test_d1.pkl"),
                         get_resource_path("./tmp/X_test_d2.pkl"),
                                           get_resource_path("./tmp/y_test.pkl"))
    test_data_size = len(test_set)
    train_data_size = len(train_set)
    train_need_batch = train_data_size / BATCH_SIZE
    test_need_batch = test_data_size / BATCH_SIZE
    logger.info("训练集大小是：{}，需要{}批完成训练".format(train_data_size, train_need_batch))
    logger.info("测试集大小是：{}，，需要{}批完成训练".format(test_data_size, test_need_batch))
    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE)
    ALPHA=train_set.get_positive_count()/len(train_set)
    logger.info("批次数为{},学习率是{},alpha为{}".format(BATCH_SIZE, RATE, ALPHA))
    if LOSS_FUN == 'focal':
        loss_fn = focal_loss(alpha=ALPHA, gamma=2, num_classes=2)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    # weight_decay=0.0001
    if OPTIM == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=RATE)
    elif OPTIM == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=RATE)
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    test_auc = []
    with open(get_resource_path("./tmp/y_train.pkl"), "rb") as f:
        y_true_train = pickle.load(f)
    with open(get_resource_path("./tmp/y_test.pkl"), "rb") as f:
        y_true_test = pickle.load(f)
    train_p = []
    train_r = []
    train_f = []
    test_p = []
    test_r = []
    test_f = []
    train_auc = []
    max_auc_epoch=-1
    max_auc = -1
    best_model = None
    for i in range(EPOCH):
        train_epoch_loss = 0  # 等于每一批次的损失值之和
        train_epoch_true = 0
        predict_proba_train = []
        epoch_train_y_predict = []
        logger.info("-------第 {} 轮训练开始-------".format(i + 1))
        logger.info("训练部分开始...")
        if ui is not None:
            ui.textBrowser_train_process.append("-------第{}轮训练开始-------\n训练部分开始...\n".format(i + 1))
            ui.textBrowser_train_process.moveCursor(ui.textBrowser_train_process.textCursor().End)
        # 只对部分图层有作用，开启dropout
        model.train()
        if if_3d:
            for data, geo_data in zip(train_dataloader, geo_loader_train):
                atom_bond_graphs, bond_angle_graphs = geo_data
                atom_bond_graphs = atom_bond_graphs.tensor()
                bond_angle_graphs = bond_angle_graphs.tensor()
                node_repr, edge_repr, graph_repr = compound_encoder(atom_bond_graphs, bond_angle_graphs)
                # 正则化
                graph_repr = norm(graph_repr)
                mol_vec,garm, targets = data
                outputs = model(mol_vec,garm, graph_repr)


                list_auc = outputs[:, 1]
                predict_proba_train.extend(list_auc.detach().numpy())
                loss = loss_fn(outputs, targets)
                train_epoch_loss = train_epoch_loss + loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                y_predict = outputs.argmax(-1)
                epoch_train_y_predict.extend(y_predict)
                accuracy_num = (y_predict == targets).sum()
                train_epoch_true += accuracy_num
        else:
            for data in train_dataloader:
                mol_vec,garm, targets = data
                outputs = model(mol_vec,garm)

                #======与3维度处理的部分一样=======
                list_auc = outputs[:, 1]
                predict_proba_train.extend(list_auc.detach().numpy())
                loss = loss_fn(outputs, targets)
                train_epoch_loss = train_epoch_loss + loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                y_predict = outputs.argmax(-1)
                epoch_train_y_predict.extend(y_predict)
                accuracy_num = (y_predict == targets).sum()
                train_epoch_true += accuracy_num
        train_epoch_acc = train_epoch_true / train_data_size
        p, r, f, s = precision_recall_fscore_support(y_true_train, epoch_train_y_predict, average=None)
        score = np.array(predict_proba_train)
        train_auc_epoch = roc_auc_score(y_true_train, score)
        train_auc.append(train_auc_epoch)
        logger.info(f"训练集已完成，当前是轮{i + 1}，训练集上的总命中个数是{train_epoch_true}/{train_data_size}，准确率"
                    f"是{train_epoch_acc}，查准率precious是{p}，查全率recall是{r}，f1是{f},AUC为{train_auc_epoch}")
        if ui is not None:
            ui.textBrowser_train_process.append(
                f"训练集已完成，当前是轮{i + 1}，训练集上的总命中个数是{train_epoch_true}/"
                f"{train_data_size}，准确率是{train_epoch_acc}，查准率precious是"
                f"{p}，查全率recall是{r}，f1是{f},"
                f"AUC为{train_auc_epoch}\n")
            ui.textBrowser_train_process.moveCursor(ui.textBrowser_train_process.textCursor().End)
        train_acc.append(train_epoch_acc.item())
        train_p.append(p)
        train_r.append(r)
        train_f.append(f)
        train_loss.append(train_epoch_loss / train_need_batch)
        test_epoch_loss = 0
        test_epoch_true = 0
        epoch_test_y_predict = []
        predict_proba_test = []
        # 测试开始
        model.eval()
        with torch.no_grad():
            logger.info("验证部分开始...")
            if ui is not None:
                ui.textBrowser_train_process.append("验证部分开始...\n")
                ui.textBrowser_train_process.moveCursor(ui.textBrowser_train_process.textCursor().End)
            if if_3d:
                for data, geo_data in zip(test_dataloader, geo_loader_test):
                    mol_vec,garm, targets = data
                    atom_bond_graphs, bond_angle_graphs = geo_data
                    atom_bond_graphs = atom_bond_graphs.tensor()
                    bond_angle_graphs = bond_angle_graphs.tensor()
                    node_repr, edge_repr, graph_repr = compound_encoder(atom_bond_graphs, bond_angle_graphs)
                    graph_repr = norm(graph_repr)
                    outputs = model(mol_vec,garm, graph_repr)
                    loss = loss_fn(outputs, targets)
                    test_epoch_loss += loss.item()
                    list_auc = outputs[:, 1]
                    predict_proba_test.extend(list_auc.numpy())
                    y_predict = outputs.argmax(-1)
                    epoch_test_y_predict.extend(y_predict)
                    test_accuracy_num = (y_predict == targets).sum()
                    test_epoch_true += test_accuracy_num
            else:
                for data in test_dataloader:
                    mol_vec,garm, targets = data
                    outputs = model(mol_vec,garm)
                    loss = loss_fn(outputs, targets)
                    test_epoch_loss += loss.item()
                    list_auc = outputs[:, 1]
                    predict_proba_test.extend(list_auc.numpy())
                    y_predict = outputs.argmax(-1)
                    epoch_test_y_predict.extend(y_predict)
                    test_accuracy_num = (y_predict == targets).sum()
                    test_epoch_true += test_accuracy_num
        test_epoch_acc = test_epoch_true / test_data_size
        p, r, f, s = precision_recall_fscore_support(y_true_test, epoch_test_y_predict, average=None)
        score = np.array(predict_proba_test)
        test_auc_epoch = roc_auc_score(y_true_test, score)
        test_auc.append(test_auc_epoch)
        logger.info(f"测试集已完成，当前是轮{i + 1}，测试集上的总命中个数是{test_epoch_true}/{test_data_size}，"
                    f"准确率是{test_epoch_acc}，查准率precious是{p}，查全率recall是{r}，f1是{f}，AUC是{test_auc_epoch}")
        if ui is not None:
            ui.textBrowser_train_process.append(
                f"测试集已完成，当前是轮{i + 1}，测试集上的总命中个数是{test_epoch_true}/{test_data_size}，"
                f"准确率是{test_epoch_acc}，查准率precious是{p}，查全率recall是{r}，f1是{f}，AUC是{test_auc_epoch}")
            ui.textBrowser_train_process.moveCursor(ui.textBrowser_train_process.textCursor().End)
        test_acc.append(test_epoch_acc.item())
        test_loss.append(test_epoch_loss / test_need_batch)
        # 存储最佳模型
        if test_auc_epoch > max_auc:
            patience = 0
            max_auc = test_auc_epoch
            max_auc_epoch=i+1
            logger.info(f"当前的auc最佳。当前轮数是{i + 1}，保存这一轮的模型")
            best_model = model
        else:
            patience+=1
        test_p.append(p)
        test_r.append(r)
        test_f.append(f)
        if patience==PATIENCE:
            print("此处早停")
            break
    logger.info(f"第{max_auc_epoch}轮的模型在验证集上的 AUC 值最高，值为：{max_auc}")
    if ui is not None:
        torch.save(best_model, os.path.join(save_model_path, "best_model.pth"))
    # 显示出折线图
    if ui is not None:
        list_x = list(range(1, len(train_auc) + 1))
        plt.figure(figsize=(6, 6))
        plt.plot(list_x, train_auc, label='训练集', color='green', alpha=0.5)
        plt.plot(list_x, test_auc, label='验证集', color='red', alpha=0.5)
        plt.legend(prop='STSong')
        plt.title("ROC-AUC的变化曲线图", fontproperties='STSong')
        plt.xlabel("轮数", fontproperties='STSong')
        plt.ylabel("曲线下面积大小", fontproperties='STSong')
        plt.tight_layout()
        plt.savefig(get_resource_path("./tmp/fig1.jpg"), pad_inches=0.1)
        plt.clf()  # 重置plt.避免上面的线也会画到下面的画布中去
        # plt.figure(figsize=(10, 6))  #这行负责新建一个画布，如果新建就不需要 clf 了
        plt.plot(list_x, train_loss, label='训练集', color='green', alpha=0.5)
        plt.plot(list_x, test_loss, label='测试集', color='red', alpha=0.5)
        plt.legend(prop='STSong', loc="best")
        plt.title("损失值变化曲线", fontproperties='STSong')
        plt.xlabel("轮数", fontproperties='STSong')
        plt.tight_layout()
        plt.savefig(get_resource_path("./tmp/fig2.jpg"), pad_inches=0.1)
        # 设置QLabel的缩放属性
        ui.label_auc.setScaledContents(True)  # 让图片自适应QLabel大小
        ui.label_loss.setScaledContents(True)
        ui.label_auc.setPixmap(QtGui.QPixmap(get_resource_path("./tmp/fig1.jpg")))
        ui.label_loss.setPixmap(QtGui.QPixmap(get_resource_path("./tmp/fig2.jpg")))
    else:
        print(model)
        print(best_model)
        return best_model


if __name__ == '__main__':
    model=train_begin(None,120,32,0.001,False)
    # 训练文件
    df = pd.read_excel(r"E:\pyqt_bioactivity\test_data\2025-2-28_newsun\test.xlsx")
    batch_predict_begin(None, df, model, False)
