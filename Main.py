import ctypes
import sys

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import *
from PyQt5 import uic
from gensim.models import word2vec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mol2vec.features import mol2alt_sentence, sentences2vec, MolSentence
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import pandas as pd
import torch
import paddle
from pahelix.model_zoo.gem_model import GeoGNNModel
from pahelix.utils import load_json_config

# 对项目自身文件的依赖
from train import abnorm, train_begin
from core_code.featurizer import DownstreamCollateFn
from core_code.d2_main import get_d2, extract_graph_train, get_d2_batch_infer
from core_code.d3_main import ectrat_d3
from core_code.d2_utils.graph_embedding import graph_embedding
from batch_predict import batch_predict_begin
from core_code.d2_utils.tools import get_resource_path
from core_code.d3_infer import ectrat_d3_infer


# 让任务栏也显示logo
myappid = "my app"
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

class MyFigure(FigureCanvasQTAgg):
    def __init__(self, width=10, height=10, dpi=100):
        #创建一个画布fig
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(MyFigure, self).__init__(self.fig)
        #添加一个子图
        self.axes = self.fig.add_subplot(111)

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.model_Path=None
        self.compounds_files=None
        self.save_model_path=None
        self.gridlayout = QGridLayout(self.ui.bar_graph_box)
        self.pos_num=None
        self.neg_num=None
        self.if_3d=False
        self.batch_size=32
        self.filePath=None
        self.save_path=None

    def init_ui(self):
        self.ui = uic.loadUi("./ui/基于多模态信息融合的药物分子活性预测系统v2.0.ui")
        # 设置logo。网上很多方法都是写作self.setWindowIcon(QIcon("E:\pyqt_bioactivity\logo.ico"))，但是这样无效，需要加".ui"
        # 更建议写作 logo.ico，这样更清晰
        # self.ui.setWindowIcon(QIcon("E:\pyqt_bioactivity\logo.png"))
        self.ui.setWindowIcon(QIcon("E:\pyqt_bioactivity\logo.ico"))

        # 绑定信号与槽函数
        self.ui.selectFileBtn.clicked.connect(self.selectFile)
        self.ui.savePathBtn.clicked.connect(self.select_savePath)
        self.ui.begin_process_btn.clicked.connect(self.begin_process)
        self.ui.choose_model_btn.clicked.connect(self.choose_model)
        self.ui.select_infer_file_btn.clicked.connect(self.select_infer_file)
        self.ui.save_infer_btn.clicked.connect(self.save_infer)
        self.ui.predictButton.clicked.connect(self.predict)
        self.ui.compounds_files_btn.clicked.connect(self.select_compounds_files)
        self.ui.save_model_path_btn.clicked.connect(self.get_save_model_path)
        self.ui.train_begin_btn.clicked.connect(self.train)
        self.ui.batch_predict_btn.clicked.connect(self.batch_predict)


    def selectFile(self):
        self.filePath, _ = QFileDialog.getOpenFileName(
            self.ui,  # 父窗口对象
            "选择要上传的分子信息文件",  # 标题
            r"./test_data/2025-2-28_newsun/train.xlsx",  # 默认路径
            "文件类型 (*.csv *.xlsx)"  # 选择类型过滤项，过滤内容在括号中
        )
        if self.filePath:
            self.ui.textBrowser_selectFile.setText("分子信息文件加载成功，它的路径是"+self.filePath)

    def choose_model(self):
        self.model_Path, _ = QFileDialog.getOpenFileName(
            self.ui,  # 父窗口对象
            "选择一个已训练模型",  # 标题
            r"",  # 起始目录
            "文件类型 (*.pth)"  # 选择类型过滤项，过滤内容在括号中
        )
        self.ui.choose_model_brower.setText("模型选择成功，它的路径是"+self.model_Path)

    #tab2中选择预处理完毕的分子文件
    def select_compounds_files(self):
        self.compounds_files, _ = QFileDialog.getOpenFileName(
            self.ui,  # 父窗口对象
            "选择预处理好的分子数据集",  # 标题
            r"",  # 起始目录
            "文件类型 (*.xlsx)"  # 选择类型过滤项，过滤内容在括号中
        )
        self.ui.textBrowser_compounds_file.setText("分子数据集选择成功，它的路径是"+self.compounds_files)
    def select_savePath(self):
        self.save_path = QFileDialog.getExistingDirectory(self.ui, "选择存储路径")
        self.ui.textBrowser_savepath.setText("预处理后的文件保存路径选择成功，存储路径为" + self.save_path)
        self.save_path=self.save_path+"//Preprocessed_compounds.xlsx"
    def get_save_model_path(self):
        self.save_model_path = QFileDialog.getExistingDirectory(self.ui, "选择模型的存储路径")
        self.ui.model_path.setText("模型的存储路径选择成功，路径为"+self.save_model_path)
    def select_infer_file(self):
        self.infer_file_path, _ = QFileDialog.getOpenFileName(
            self.ui,  # 父窗口对象
            "选择一个需要批量处理的文件(支持xlsx和csv后缀)",  # 标题
            r"./test_data/2025-2-28_newsun/test.xlsx",  # 默认路径
            "文件类型 (*.csv *.xlsx)"  # 选择类型过滤项，过滤内容在括号中
        )
        if self.infer_file_path:
            self.ui.textBrowser_infer_file.setText("待预测的分子文件加载成功，它的路径是"+self.infer_file_path)
    def save_infer(self):
        self.save_infer_result = QFileDialog.getExistingDirectory(self.ui, "选择存储预测结果的路径")
        self.ui.textBrowser_save_infer.setText("预测后的文件保存路径选择成功，存储路径为" + self.save_infer_result+"/predict_result.xlsx")
        self.save_infer_result=self.save_infer_result+"//predict_result.xlsx"

    #点击“开始处理”按钮
    def begin_process(self):
        #读取到的ic_min和ic_max是str类型，要转换为int类型
        ic_min = int(self.ui.ic50_min.text())
        ic_max = int(self.ui.ic50_max.text())

        if ic_min==ic_max:
            msg= "当前的分子文件来源于" + self.filePath + "\nIC50阈值设置完毕，阈值为" + str(ic_max) + "。\n开始预处理!!"
        else:
            msg= "当前的分子文件来源于" + self.filePath + "\nIC50阈值设置完毕。值小于" + str(ic_min) + "的视为活性分子，大于" + str(
                ic_max) + "则视为非活性分子。介于二者之间的为中间状态，将被舍弃。\n开始预处理!!"
        self.ui.processing_text.setText(msg)
        #判断路径是什么后缀
        if self.filePath.endswith('xlsx'):
            compounds = pd.read_excel(self.filePath)
        else:
            compounds = pd.read_csv(self.filePath)
        #文件中最初的分子数目
        init_compounds_num=len(compounds)
        #去除IC50为空的列
        compounds_IC50notNULL = compounds[compounds.standard_value.notna()]
        compounds_IC50notNULL['standard_value'] = compounds_IC50notNULL['standard_value'].astype(float)
        #增加一列bioactivity_class
        if ic_min == ic_max:
            compounds_IC50notNULL['bioactivity_class'] = compounds_IC50notNULL['standard_value'].map(
                # lambda x: 1 if x <=(1000*ic_min) else 0)
                lambda x: 1 if x <= (ic_min) else 0)
        else:
            compounds_IC50notNULL['bioactivity_class'] = compounds_IC50notNULL['standard_value'].map(
                # lambda x: 1 if x <= 1000*ic_min else (0 if x >=1000*ic_max else 'intermediate'))
                lambda x: 1 if x <=ic_min else (0 if x >= ic_max else 'intermediate'))
            compounds_IC50notNULL = compounds_IC50notNULL[compounds_IC50notNULL['bioactivity_class'] != 'intermediate']
        #只保留会用到的列，即smiles和活性值两列
        selection = ['canonical_smiles', 'bioactivity_class']
        df = compounds_IC50notNULL[selection]
        #inplace是否在原对象基础上进行修改。True表示直接修改原对象
        df.canonical_smiles.replace('nan', np.nan, inplace=True)
        # 只要有一列有nan就删除这一行
        df.dropna(inplace=True)
        #去重，重新排列索引
        df.reset_index(inplace=True, drop=True)
        # 给两个列均重命名。
        df.columns = ['smiles', 'label']
        # 去除异常分子
        df=abnorm(df)
        #存储处理后文件
        df.to_excel(self.save_path)
        #预处理后，剩余的分子数目
        compounds_num=len(df)
        self.ui.processing_text.setText(msg+"\n............\n数据处理完成！！\n已将结果保存至"+self.save_path)
        positive = df[df['label'] == 1]
        self.ui.textBrowser_processResult.setText("加载的文件中共有分子{}个。\n\n去除重复分子、关键字段为空、无法提取二维数据的分子以及舍弃中间状态的分子（仅当IC50的上下阈值不相等时）后，还剩下{"
                                                  "}个。\n\n其中活性分子有{"
                                                  "}个，非活性有{}个。\n\n"
                                                  "分子数据集活性情况的柱形图如右侧所示-->".format(
            init_compounds_num,compounds_num,len(positive),compounds_num-len(positive)))
        #显示出分子数据集活性分类柱形图
        self.F = MyFigure(width=2, height=2, dpi=100)
        self.F.axes.bar(x=[0, 1], height=[compounds_num - len(positive), len(positive)], color=['skyblue',
                                                                                                'orange'])
        self.F.axes.set_xticks([0,1])
        self.F.axes.set_xticklabels(["非活性","活性"],fontproperties="STSong")
        self.F.axes.set_ylabel('数量', fontsize=9, fontweight='bold', fontproperties="STSong")
        self.gridlayout.addWidget(self.F, 0, 0)

    #点击“开始训练”按钮后
    def train(self):
        #获取界面中输入的三种训练参数
        epoch=self.ui.spinBox.value()
        self.batch_size = self.ui.spinBox_2.value()
        lr= self.ui.doubleSpinBox.value()
        # # 设置默认参数值  （已经在 pyqt 程序中设置了默认值，原先未设置时三个值都为 0 ，设置后这里不用判断是否为0了）
        # if epoch==0:
        #     epoch=150
        # if batch_size==0:
        #     batch_size=32
        # if lr==0:
        #     lr=0.008
        train_msg="训练参数设置为，训练轮数：{}；批数：{}；学习率：{}。\n以9：1的比例划分训练集与验证集。".format(epoch,self.batch_size,lr)
        self.ui.textBrowser_train_process.setText(train_msg)
        #读入分子文件
        df= pd.read_excel(self.compounds_files)
        backup_df=df
        #一维特征提取
        df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
        df['ecfp'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)
        model = word2vec.Word2Vec.load(get_resource_path('./Dependencies/model_300dim.pkl'))
        list_mol = []
        for x in sentences2vec(df['ecfp'], model, unseen='UNK'):
            list_mol.append(x)
        label_list = df['label'].to_list()
        #划分数据集
        X_train, X_test, y_train, y_test = train_test_split(list_mol, label_list, test_size=0.1, stratify=label_list,
                                                            random_state=27)
        train_msg=train_msg+"训练集大小是：{}，测试集大小是{}。\n正在提取一维层面分子特征......\n".format(len(y_train),len(y_test))
        self.ui.textBrowser_train_process.setText(train_msg)
        with open(get_resource_path("./tmp/X_train_d1.pkl"), "wb") as f:
            pickle.dump(X_train, f)
        with open(get_resource_path("./tmp/X_test_d1.pkl"), "wb") as f:
            pickle.dump(X_test, f)
        with open(get_resource_path("./tmp/y_train.pkl"), "wb") as f:
            pickle.dump(y_train, f)
        with open(get_resource_path("./tmp/y_test.pkl"), "wb") as f:
            pickle.dump(y_test, f)
        #二维特征提取
        train_msg = train_msg + "一维特征提取完毕\n。正在提取分子二维层面的特征......\n【可能需要较久时间，敬请耐心等候！】"
        self.ui.textBrowser_train_process.setText(train_msg)
        #提取分子特征
        get_d2(backup_df,backup_df['label'])
        train_msg = train_msg + "二维特征提取完毕\n"
        self.ui.textBrowser_train_process.setText(train_msg)
        #三维特征提取
        if self.if_3d:
            train_msg = train_msg + "正在提取分子三维层面的特征，请稍后\n\n【可能需要较久时间，敬请耐心等候！】"
            self.ui.textBrowser_train_process.setText(train_msg)
            ectrat_d3(self.compounds_files,backup_df['label'])
            train_msg = train_msg + "三维特征提取完毕！\n"
            self.ui.textBrowser_train_process.setText(train_msg)
        #训练模型
        train_msg = train_msg + "开始训练模型！\n"
        self.ui.textBrowser_train_process.setText(train_msg)
        train_begin(self.ui,epoch,self.batch_size,lr,self.if_3d,self.save_model_path)

    # 点击“预测”按钮之后
    def predict(self):
        # smiles=self.ui.smiles_text.text()#单行文本框使用text获取内容，多行文本框则是用toPlainText
        smiles = self.ui.smiles_text.toPlainText()
        # 一维层面的预处理
        mol = Chem.MolFromSmiles(smiles)
        df = pd.DataFrame({'smiles': [smiles]})
        df['sentence'] = df.apply(lambda x: MolSentence(mol2alt_sentence(mol, 1)), axis=1)
        model = word2vec.Word2Vec.load(get_resource_path('./Dependencies/model_300dim.pkl'))
        vec1 = sentences2vec(df['sentence'], model, unseen='UNK')
        # 二维层面预处理
        morgan_fps = []
        mol = Chem.MolFromSmiles(smiles)
        fingerprints = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        morgan_fps.append(fingerprints.ToBitString())
        # data_pd = [{"smiles": smiles, "Fingerprints":morgan_fps}]
        extract_graph_train(out_file_path=get_resource_path("./tmp/infer_d2feature.npz"),
                            is_infer=True, smiles_list=[smiles],max_atom_num=len(mol.GetAtoms()))
        vec2 = graph_embedding(weight_path=get_resource_path("./tmp/CBoW_50dim.pt"), infer=True,infer_path=get_resource_path("./tmp/infer_d2feature.npz"))
        # 加载模型
        model = torch.load(self.model_Path)
        model.eval()
        # 使用模型预测
        vec1 = torch.Tensor(vec1)
        vec1 = vec1.unsqueeze(0)
        vec2 = torch.Tensor(vec2)
        vec2 = vec2.unsqueeze(0)
        if self.if_3d:
            # 三维层面预处理
            vec3 = ectrat_d3_infer(smiles)
            # 加载GNN
            compound_encoder_path = get_resource_path("./Dependencies/class.pdparams")
            compound_encoder_config = load_json_config(get_resource_path("./Dependencies/geognn2.json"))
            compound_encoder = GeoGNNModel(compound_encoder_config)
            compound_encoder.set_state_dict(paddle.load(compound_encoder_path))
            # 创建类的实例
            collate_fn = DownstreamCollateFn(
                # 原子属性、边属性
                atom_names=compound_encoder_config['atom_names'],
                bond_names=compound_encoder_config['bond_names'],
                # 只有一个属性，键长
                bond_float_names=compound_encoder_config['bond_float_names'],
                # 仅有一个属性，键角
                bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
                task_type='class', is_inference=True)
            # 使用这个类的方法
            geo_data = collate_fn.__call__(vec3)
            atom_bond_graphs, bond_angle_graphs = geo_data
            atom_bond_graphs = atom_bond_graphs.tensor()
            bond_angle_graphs = bond_angle_graphs.tensor()
            node_repr, edge_repr, graph_repr = compound_encoder(atom_bond_graphs, bond_angle_graphs)
            norm = paddle.nn.LayerNorm(compound_encoder.graph_dim)
            graph_repr = norm(graph_repr)
            outputs = model(vec1,vec2, graph_repr)
        else:
            outputs = model(vec1,vec2)
        softmax = torch.nn.Softmax()
        outputs = softmax(outputs)
        y_predict = outputs.argmax(-1)
        outputs = outputs.tolist()
        if y_predict.item() == 1:
            str = "活性"
        else:
            str = "非活性"
        self.ui.textBrowser_predict_result.setText(
            "分子预测完毕！\n它为非活性的概率是{:.5f}，为活性的概率是{:.5f}\n最终的预测结果是{}".format(
                outputs[0][0], outputs[0][1], str))

    def batch_predict(self):
        # 读入分子文件
        df = pd.read_excel(self.infer_file_path)
        predict_all=batch_predict_begin(self.ui,df,self.model_Path,self.if_3d)
        result_df = pd.DataFrame({
            'smiles': df['smiles'],
            'pred': predict_all
        })
        result_df.to_excel(self.save_infer_result, index=True)
        result_text = "分子预测完毕！\n\n"
        result_text += "预测结果汇总：\n"
        result_text += "SMILES\t\t预测结果\n"
        result_text += "-" * 50 + "\n"
        for idx, row in result_df.iterrows():
            result_text += f"{row['smiles']}\t{row['pred']}\n"
        self.ui.textBrowser_infer_info.setText(result_text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWindow()
    w.ui.show()
    app.exec()