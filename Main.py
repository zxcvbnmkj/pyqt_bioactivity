import ctypes
import sys
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import *
from PyQt5 import uic
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import torch
from infer import batch_predict_begin, predict_single
from train import abnorm, extract_feat, train_begin

# 让任务栏也显示logo
myappid = "my app"
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)


class MyFigure(FigureCanvasQTAgg):
    def __init__(self, width=10, height=10, dpi=100):
        # 创建一个画布fig
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(MyFigure, self).__init__(self.fig)
        # 添加一个子图
        self.axes = self.fig.add_subplot(111)


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.model_Path = None
        self.compounds_files = None
        self.save_model_path = None
        self.gridlayout = QGridLayout(self.ui.bar_graph_box)
        self.pos_num = None
        self.neg_num = None
        self.if_3d = False
        self.batch_size = 32
        self.filePath = None
        self.save_path = None
        self.my_dataset = "tmp"
        self.if_cuda = False
        self.device = torch.device('cuda' if self.if_cuda and torch.cuda.is_available() else 'cpu')

    def init_ui(self):
        self.ui = uic.loadUi("./ui/基于多模态信息融合的药物分子活性预测系统v3.0.ui")
        # 设置logo。网上很多方法都是写作self.setWindowIcon(QIcon("E:\pyqt_bioactivity\logo.ico"))，但是这样无效，需要加".ui"
        # 更建议写作 logo.ico，这样更清晰
        self.ui.setWindowIcon(QIcon("E:\pyqt_bio_v3.0\logo.ico"))

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
            self.ui.textBrowser_selectFile.setText("分子信息文件加载成功，它的路径是" + self.filePath)

    def choose_model(self):
        self.model_Path, _ = QFileDialog.getOpenFileName(
            self.ui,  # 父窗口对象
            "选择一个已训练模型",  # 标题
            r"",  # 起始目录
            "文件类型 (*.pth)"  # 选择类型过滤项，过滤内容在括号中
        )
        self.ui.choose_model_brower.setText("模型选择成功，它的路径是" + self.model_Path)

    # tab2中选择预处理完毕的分子文件
    def select_compounds_files(self):
        self.compounds_files, _ = QFileDialog.getOpenFileName(
            self.ui,  # 父窗口对象
            "选择预处理好的分子数据集",  # 标题
            r"",  # 起始目录
            "文件类型 (*.xlsx)"  # 选择类型过滤项，过滤内容在括号中
        )
        self.ui.textBrowser_compounds_file.setText("分子数据集选择成功，它的路径是" + self.compounds_files)

    def select_savePath(self):
        self.save_path = QFileDialog.getExistingDirectory(self.ui, "选择存储路径")
        self.ui.textBrowser_savepath.setText("预处理后的文件保存路径选择成功，存储路径为" + self.save_path)
        self.save_path = self.save_path + "//Preprocessed_compounds.xlsx"

    def get_save_model_path(self):
        self.save_model_path = QFileDialog.getExistingDirectory(self.ui, "选择模型的存储路径")
        self.ui.model_path.setText("模型的存储路径选择成功，路径为" + self.save_model_path)

    def select_infer_file(self):
        self.infer_file_path, _ = QFileDialog.getOpenFileName(
            self.ui,  # 父窗口对象
            "选择一个需要批量处理的文件(支持xlsx和csv后缀)",  # 标题
            r"./test_data/2025-2-28_newsun/test.xlsx",  # 默认路径
            "文件类型 (*.csv *.xlsx)"  # 选择类型过滤项，过滤内容在括号中
        )
        if self.infer_file_path:
            self.ui.textBrowser_infer_file.setText("待预测的分子文件加载成功，它的路径是" + self.infer_file_path)

    def save_infer(self):
        self.save_infer_result = QFileDialog.getExistingDirectory(self.ui, "选择存储预测结果的路径")
        self.ui.textBrowser_save_infer.setText(
            "预测后的文件保存路径选择成功，存储路径为" + self.save_infer_result + "/predict_result.xlsx")
        self.save_infer_result = self.save_infer_result + "//predict_result.xlsx"

    # 点击“开始处理”按钮
    def begin_process(self):
        # 读取到的ic_min和ic_max是str类型，要转换为int类型
        ic_min = int(self.ui.ic50_min.text())
        ic_max = int(self.ui.ic50_max.text())

        if ic_min == ic_max:
            msg = "当前的分子文件来源于" + self.filePath + "\nIC50阈值设置完毕，阈值为" + str(ic_max) + "。\n开始预处理!!"
        else:
            msg = "当前的分子文件来源于" + self.filePath + "\nIC50阈值设置完毕。值小于" + str(
                ic_min) + "的视为活性分子，大于" + str(
                ic_max) + "则视为非活性分子。介于二者之间的为中间状态，将被舍弃。\n开始预处理!!"
        self.ui.processing_text.setText(msg)
        # 判断路径是什么后缀
        if self.filePath.endswith('xlsx'):
            compounds = pd.read_excel(self.filePath)
        else:
            compounds = pd.read_csv(self.filePath)
        # 文件中最初的分子数目
        init_compounds_num = len(compounds)
        # 去除IC50为空的列
        compounds_IC50notNULL = compounds[compounds.standard_value.notna()]
        compounds_IC50notNULL['standard_value'] = compounds_IC50notNULL['standard_value'].astype(float)
        # 增加一列bioactivity_class
        if ic_min == ic_max:
            compounds_IC50notNULL['bioactivity_class'] = compounds_IC50notNULL['standard_value'].map(
                # lambda x: 1 if x <=(1000*ic_min) else 0)
                lambda x: 1 if x <= (ic_min) else 0)
        else:
            compounds_IC50notNULL['bioactivity_class'] = compounds_IC50notNULL['standard_value'].map(
                # lambda x: 1 if x <= 1000*ic_min else (0 if x >=1000*ic_max else 'intermediate'))
                lambda x: 1 if x <= ic_min else (0 if x >= ic_max else 'intermediate'))
            compounds_IC50notNULL = compounds_IC50notNULL[compounds_IC50notNULL['bioactivity_class'] != 'intermediate']
        # 只保留会用到的列，即smiles和活性值两列
        selection = ['canonical_smiles', 'bioactivity_class']
        df = compounds_IC50notNULL[selection]
        # inplace是否在原对象基础上进行修改。True表示直接修改原对象
        df.canonical_smiles.replace('nan', np.nan, inplace=True)
        # 只要有一列有nan就删除这一行
        df.dropna(inplace=True)
        # 去重，重新排列索引
        df.reset_index(inplace=True, drop=True)
        # 给两个列均重命名。
        df.columns = ['smiles', 'label']
        # 去除异常分子
        df=abnorm(df)
        # 存储处理后文件
        df.to_excel(self.save_path)
        # 预处理后，剩余的分子数目
        compounds_num = len(df)
        self.ui.processing_text.setText(msg + "\n............\n数据处理完成！！\n已将结果保存至" + self.save_path)
        positive = df[df['label'] == 1]
        self.ui.textBrowser_processResult.setText(
            "加载的文件中共有分子{}个。\n\n去除重复分子、关键字段为空、无法提取二维数据的分子以及舍弃中间状态的分子（仅当IC50的上下阈值不相等时）后，还剩下{"
            "}个。\n\n其中活性分子有{"
            "}个，非活性有{}个。\n\n"
            "分子数据集活性情况的柱形图如右侧所示-->".format(
                init_compounds_num, compounds_num, len(positive), compounds_num - len(positive)))
        # 显示出分子数据集活性分类柱形图
        self.F = MyFigure(width=2, height=2, dpi=100)
        self.F.axes.bar(x=[0, 1], height=[compounds_num - len(positive), len(positive)], color=['skyblue',
                                                                                                'orange'])
        self.F.axes.set_xticks([0, 1])
        self.F.axes.set_xticklabels(["非活性", "活性"], fontproperties="STSong")
        self.F.axes.set_ylabel('数量', fontsize=9, fontweight='bold', fontproperties="STSong")
        self.gridlayout.addWidget(self.F, 0, 0)

    # 点击“开始训练”按钮后
    def train(self):
        # 获取界面中输入的三种训练参数
        epoch = self.ui.spinBox.value()
        self.batch_size = self.ui.spinBox_2.value()
        lr = self.ui.doubleSpinBox.value()
        alpha = self.ui.doubleSpinBox_alpha.value()
        self.if_cuda = self.ui.checkBox.isChecked()
        self.device = torch.device('cuda' if self.if_cuda and torch.cuda.is_available() else 'cpu')
        train_msg = "当前使用的训练设备是：{}".format(self.device)
        self.ui.textBrowser_train_process.setText(train_msg)
        # //是整除
        if epoch % 3 != 0:
            epoch = ((epoch + 2) // 3) * 3
        train_msg = "训练参数设置为，训练轮数：{}；批数：{}；学习率：{}。\n开始提取三个维度的特征，需要较久时间，请耐心等候！\n".format(
            epoch, self.batch_size, lr)
        self.ui.textBrowser_train_process.setText(train_msg)
        extract_feat(my_dataset=self.my_dataset, if_infer=False, device=self.if_cuda,
                     compounds_files=self.compounds_files)
        # 训练模型
        train_msg = train_msg + "特征提取已结束。\n开始训练模型！\n"
        self.ui.textBrowser_train_process.setText(train_msg)
        train_begin(ui=self.ui, epoch=epoch, batch_size=self.batch_size, lr=lr, save_model_path=self.save_model_path,
                    my_dataset=self.my_dataset, device=self.device, ALPHA=alpha)

    # 点击“预测”按钮之后
    def predict(self):
        # smiles=self.ui.smiles_text.text()#单行文本框使用text获取内容，多行文本框则是用toPlainText
        smiles = self.ui.smiles_text.toPlainText()
        print("获取到的文本内容是", smiles)
        result = predict_single(smiles=smiles, my_dataset=self.my_dataset, device=self.device,
                                save_model_path=self.model_Path)
        self.ui.textBrowser_predict_result.setText(
            "分子预测完毕！\n它为非活性的概率是{:.5f}，为活性的概率是{:.5f}\n最终的预测结果是: {}".format(
                result[0][0][0], result[0][0][1], result[1][0]))

    def batch_predict(self):
        if self.ui is not None:
            infer_msg = "批量预测开始！\n"
            self.ui.textBrowser_infer_info.setText(infer_msg)

        predict_all = batch_predict_begin(my_dataset=self.my_dataset, device=self.device,
                                          compounds_files=self.infer_file_path, batch_size=self.batch_size,
                                          save_model_path=self.model_Path)

        df = pd.read_excel(self.infer_file_path)
        result_df = pd.DataFrame({
            'smiles': df['smiles'],
            'pred': predict_all[1]
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
    os.system("pause")
