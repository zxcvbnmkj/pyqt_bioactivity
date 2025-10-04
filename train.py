from matplotlib import pyplot as plt
from rdkit.Chem import AllChem
import logging
import os
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import paddle
from pahelix.model_zoo.gem_model import GeoGNNModel
from pahelix.utils import load_json_config
from train_util.Focal_Loss import focal_loss
from train_util.fusion import History, model_forward
from train_util.model import theModel_final
from train_util.dataset import MYDataset
from train_util.featurizer import DownstreamCollateFn, DownstreamTransformFn
from train_util.utils_d3 import _save_npz_data, load_bioactivity_dataset
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, MolSentence
from rdkit import Chem
import pickle
from train_util.utils import get_2d
import pandas as pd
from pahelix.datasets import InMemoryDataset
from pyqt5_plugins.examplebuttonplugin import QtGui
import warnings

warnings.filterwarnings("ignore")


# 不需要从conda环境中去修改mol2vec的代码，其中的一个函数中有个语句需要改，最简单的做法其实是在自己的文件中重写该函数，在调用时，不调用库
# 里的函数，而调用重写的函数。
def fix_sentences2vec(sentences, model, unseen=None):
    # keys = set(model.wv.vocab.keys())
    keys = set(model.wv.index_to_key)
    vec = []
    if unseen:
        unseen_vec = model.wv.word_vec(unseen)

    for sentence in sentences:
        if unseen:
            vec.append(sum([model.wv.word_vec(y) if y in set(sentence) & keys
                            else unseen_vec for y in sentence]))
        else:
            vec.append(sum([model.wv.word_vec(y) for y in sentence
                            if y in set(sentence) & keys]))
    return np.array(vec)


def create_logger(my_dataset):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(
        filename=f"./{my_dataset}_Intermediate/train.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger


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


def extract_feat(my_dataset="tmp", if_infer=False, device="cpu", compounds_files=None):
    if not os.path.exists(f"./{my_dataset}_Intermediate"):
        os.makedirs(f"./{my_dataset}_Intermediate/")
    df = pd.read_excel(compounds_files)
    model = word2vec.Word2Vec.load('./Dependencies/model_300dim.pkl')
    if not if_infer:
        # 训练集一维提取
        # smiles trans to ecfp
        df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
        df['ecfp'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)
        # Training or using pre trained molecular word embedding models
        list_mol = []
        for x in fix_sentences2vec(df['ecfp'], model, unseen='UNK'):
            list_mol.append(x)
        label_list = df['label'].to_list()
        with open(f"./{my_dataset}_Intermediate/X_train_d1.pkl", "wb") as f:
            pickle.dump(list_mol, f)
        with open(f"./{my_dataset}_Intermediate/y_train.pkl", "wb") as f:
            pickle.dump(label_list, f)
    else:
        # 测试集一维提取。没有y轴标签
        print("The total size of this dataset is ", len(df))
        df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
        df['ecfp'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)
        list_mol = []
        for x in fix_sentences2vec(df['ecfp'], model, unseen='UNK'):
            list_mol.append(x)
        with open(f"./{my_dataset}_Intermediate/X_test_d1.pkl", "wb") as f:
            pickle.dump(list_mol, f)
    if not if_infer:
        get_2d(data_pd=df, my_dataset=my_dataset, max_atoms=100, device=device, if_infer=False)
    else:
        get_2d(data_pd=df, my_dataset=my_dataset, max_atoms=100, device=device, if_infer=True)
    # 提取三维
    dataset = load_bioactivity_dataset(compounds_files)
    dataset.transform(DownstreamTransformFn(), num_workers=1)
    # The default storage flie is part1.npz
    dataset.save_data(f"./{my_dataset}_Intermediate/")
    data = dataset._load_npz_data_files([f"./{my_dataset}_Intermediate/part1.npz"])
    if not if_infer:
        _save_npz_data(data, f"./{my_dataset}_Intermediate/X_train_d3.npz")
    else:
        _save_npz_data(data, f"./{my_dataset}_Intermediate/X_test_d3.npz")


def train_begin(ui, epoch=60, batch_size=32, lr=0.004, save_model_path=None, my_dataset="tmp", device="cpu", ALPHA=0.5):
    LOSS_FUN = "focal"
    MY_SCHEDULER = True
    OPTIM = 'Adam'
    PATIENCE = 20
    compound_encoder_path = "./train_util/class.pdparams"
    logger = create_logger(my_dataset)

    compound_encoder_config = load_json_config("./train_util/gnnconfig.json")
    # The GeoGNNModel model of the PaddlePaddle platform is used here to process the three-dimensional molecular information
    compound_encoder = GeoGNNModel(compound_encoder_config)
    compound_encoder.set_state_dict(paddle.load(compound_encoder_path))
    train_dataset = InMemoryDataset(
        npz_data_files=[f"./{my_dataset}_Intermediate/X_train_d3.npz"])
    collate_fn = DownstreamCollateFn(
        atom_names=compound_encoder_config['atom_names'],
        bond_names=compound_encoder_config['bond_names'],
        bond_float_names=compound_encoder_config['bond_float_names'],
        bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
        task_type='class', is_inference=True)
    geo_loader_train = train_dataset.get_data_loader(
        batch_size=batch_size,
        num_workers=1,
        collate_fn=collate_fn)
    model = theModel_final()
    # whether or not use cuda（1）
    if device == "cuda":
        model.cuda()
        print("The CUDA device currently in use is device ", {torch.cuda.current_device()})
    train_set = MYDataset(f"./{my_dataset}_Intermediate/X_train_d1.pkl",
                          f"./{my_dataset}_Intermediate/X_train_d2.pkl",
                          f"./{my_dataset}_Intermediate/y_train.pkl")
    train_data_size = len(train_set)
    train_need_batch = train_data_size / batch_size
    # 如果用户没有指定ALPHA的值
    if ALPHA == 0.0:
        # -0.008是因为该任务一般会比较关注活性，alpha表示的是对非活性的权重
        ALPHA = train_set.get_positive_count() / len(train_set) - 0.008
    logger.info("batch_size:{},lr:{},alpha:{}".format(batch_size, lr, ALPHA))
    logger.info("The trainset size is {}, and a {} batch is required to complete the training".format(train_data_size,
                                                                                                      train_need_batch))
    if ui is not None:
        ui.textBrowser_train_process.append(
            "轮次是:{},批次是:{},学习率是:{},alpha值是:{},训练数据大小是{}".format(epoch, batch_size, lr, ALPHA,
                                                                                   train_data_size))
        ui.textBrowser_train_process.moveCursor(ui.textBrowser_train_process.textCursor().End)
    train_dataloader = DataLoader(train_set, batch_size=batch_size)
    if LOSS_FUN == 'focal':
        loss_fn = focal_loss(alpha=ALPHA, gamma=2, num_classes=2)
        loss_fn2 = focal_loss(alpha=ALPHA, gamma=2, num_classes=2, size_average="BatchSize")
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        loss_fn2 = torch.nn.CrossEntropyLoss(reduction=None)
    # weight_decay=0.0001
    if OPTIM == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif OPTIM == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if MY_SCHEDULER == True:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch / 3, eta_min=0.0005)
    # read label
    with open(f"./{my_dataset}_Intermediate/y_train.pkl", "rb") as f:
        y_true_train = pickle.load(f)
    norm = paddle.nn.LayerNorm(compound_encoder.graph_dim)

    d1_history = History(len(train_dataloader.dataset), device="cpu")
    d3_history = History(len(train_dataloader.dataset), device="cpu")
    gram_history = History(len(train_dataloader.dataset), device="cpu")
    # The definitions of these variables must be placed before the start of training
    max_auc = 0
    patience = 0
    train_auc = []
    train_loss = []
    best_model = None
    for i in range(epoch):
        train_epoch_loss = 0
        train_epoch_true = 0
        predict_proba_train = []
        epoch_train_y_predict = []
        logger.info("------- Training for round {} begins -------".format(i + 1))
        logger.info("> > > >train< < < <")
        if ui is not None:
            ui.textBrowser_train_process.append(
                "正在进行第{}轮训练".format(i + 1))
            ui.textBrowser_train_process.moveCursor(ui.textBrowser_train_process.textCursor().End)
        model.train()
        for data, geo_data in zip(train_dataloader, geo_loader_train):
            atom_bond_graphs, bond_angle_graphs = geo_data
            node_repr, edge_repr, graph_repr = compound_encoder(atom_bond_graphs.tensor(), bond_angle_graphs.tensor())
            graph_repr = norm(graph_repr)
            gram, mol_vec, targets, index = data
            loss, outputs, _ = model_forward(model, loss_fn, loss_fn2, gram, mol_vec, graph_repr, targets, index,
                                             d1_history, d3_history, gram_history, mode="train", device="cpu")
            if device == "cuda":
                outputs = outputs.cpu().detach()
            list_auc = outputs[:, 1]
            predict_proba_train.extend(list_auc.detach().numpy())
            train_epoch_loss = train_epoch_loss + loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if MY_SCHEDULER == True:
                scheduler.step()
            y_predict = outputs.argmax(-1)
            epoch_train_y_predict.extend(y_predict)
            accuracy_num = (y_predict == targets).sum()
            train_epoch_true += accuracy_num
        train_epoch_acc = train_epoch_true / train_data_size
        p, r, f, s = precision_recall_fscore_support(y_true_train, epoch_train_y_predict, average=None)
        score = np.array(predict_proba_train)
        train_auc_epoch = roc_auc_score(y_true_train, score)
        train_auc.append(train_auc_epoch)
        train_loss.append(train_epoch_loss)
        logger.info(
            f"Training set completed，The current number of epoch is {i + 1}，The total number of hits on the training set is {train_epoch_true}/{train_data_size}，acc"
            f" is {train_epoch_acc}，precious is {p}，recall is {r}，f1 is {f},AUC is {train_auc_epoch}")
        if ui is not None:
            ui.textBrowser_train_process.append(
                f"本轮训练完毕，命中数量是{train_epoch_true}/{train_data_size},准确率是{train_epoch_acc},精确率是{p}，召回率是{r},f1值是{f},AUC值是{train_auc_epoch}".format(
                    i + 1))
            ui.textBrowser_train_process.moveCursor(ui.textBrowser_train_process.textCursor().End)
        # macro
        # p, r, f, s = precision_recall_fscore_support(y_true_train, epoch_train_y_predict, average="macro")
        # logger.info(f"macro precious is {p}，macro recall is {r}，macro f1 is {f}")
        # cm = confusion_matrix(y_true_train, epoch_train_y_predict, normalize='true')
        # logger.info(f"The train set confusion matrix is:\n{cm}")

        if train_auc_epoch > max_auc:
            patience = 0
            max_auc = train_auc_epoch
            logger.info(
                f"The current AUC is the best. The current number of rounds is {i + 1}, store the model for this round")
            best_model = model
        else:
            patience += 1
            logger.info(f"The current tolerance count is {patience}")
        if patience == PATIENCE:
            logger.info(f"Continuous {patience} epochs of AUC without improvement, stop training")
            break
    logger.info(f"The maximum AUC value is {max_auc}")
    torch.save(best_model, f"{save_model_path}/best_model.pth")
    # 显示出折线图
    if ui is not None:
        list_x = list(range(1, len(train_auc) + 1))
        plt.figure(figsize=(6, 6))
        plt.plot(list_x, train_auc, label='训练集', color='green', alpha=0.5)
        plt.title("ROC-AUC的变化曲线图", fontproperties='STSong')
        plt.xlabel("轮数", fontproperties='STSong')
        plt.ylabel("曲线下面积大小", fontproperties='STSong')
        plt.tight_layout()
        plt.savefig(f"./{my_dataset}_Intermediate/fig1.jpg", pad_inches=0.1)
        plt.clf()  # 重置plt.避免上面的线也会画到下面的画布中去
        # plt.figure(figsize=(10, 6))  #这行负责新建一个画布，如果新建就不需要 clf 了
        plt.plot(list_x, train_loss, label='训练集', color='green', alpha=0.5)
        plt.title("损失值变化曲线", fontproperties='STSong')
        plt.xlabel("轮数", fontproperties='STSong')
        plt.tight_layout()
        plt.savefig(f"./{my_dataset}_Intermediate/fig2.jpg", pad_inches=0.1)
        # 设置QLabel的缩放属性
        ui.label_auc.setScaledContents(True)  # 让图片自适应QLabel大小
        ui.label_loss.setScaledContents(True)
        ui.label_auc.setPixmap(QtGui.QPixmap(f"./{my_dataset}_Intermediate/fig1.jpg"))
        ui.label_loss.setPixmap(QtGui.QPixmap(f"./{my_dataset}_Intermediate/fig2.jpg"))
    else:
        print(model)
        print(best_model)
        return best_model


if __name__ == '__main__':
    extract_feat(compounds_files=r"E:\pyqt_bio_v3.0\test_data\2025-2-28_newsun\Preprocessed_compounds.xlsx")
    model = train_begin(None, 150, 32, 0.001, "tmp2_Intermediate")
