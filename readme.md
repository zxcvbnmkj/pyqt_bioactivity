# News 📢 : V3.0 Released! (2025.10.04) 🎉
# 药物分子活性预测 PYQT 程序 V3.0 使用手册
## Guide
1. 程序运行时，如果需要写入内容到一个 excel 表格中，但该表格在电脑处于被打开状态，将导致写入异常，从而引起程序闪退。请务必保证 excel 表格及时关闭。
2. 重要的超参数 `alpha`。这个参数等于数据集中活性样本所占比例，即活性样本数量除以数据总量。

该参数的值越大，说明数据集中活性样本最多。若数据集中活性、非活性样本个数一样多，则 `aplha = 0.5` 。

若值大于 `0.5` ，说明活性样本多于非活性，为了避免数据不平衡，在训练时模型就会更关注非活性样本是否被预测正确，使得在训练完的模型推理时，更偏向预测分子为非活性的。反之，小于 `0.5` ，模型会更关注活性样本。

因此，若发现了预测时，分子总是偏向阴性，可以尝试把alpha的值设置小

## Updata 
- [x] 令用户可自行调整关键超参数 `alpha` ，程序中该值默认为 0 ，意为通过训练集正负样本比例自动计算。若此处值不为 0 ，则认为用户进行了自定义，模型训练阶段将使用自定义值。
- [x] 为了提示训练效果，增加了“学习率调度器” `Scheduler` 。本项目采用了“余弦退火”的调度策略，并设置了 `T_max=epoch / 3`。这导致 epoch 需要是 3 的倍数，若不是，代码将其重置为了与原数最接近的 epoch 。
- [x] `requirements.txt` 中已经列了本项目所需的全部依赖，能够一次性部署好环境。

## 记录
1. 二维特征提取中的`node_embedding`的时候，不依赖于测试集和验证集，完全可以省略。
2. 因为数据比较难收集，避免浪费，不设置验证集，在所有数据上训练。
3. 在 pyqt 程序中如果遇到报错，可能只出现 `Process finished with exit code -1073740791 (0xC0000409)` 而不显示具体报错行数和原因。进程异常退出，其实是因为代码错误导致的，只要使用调试即可找到是哪一行代码的问题。但是更建议避开运行程序，直接在代码中调用事件函数来测试代码是否有错。

## 奇怪的问题（已解决，但原因未知）
在程序中连按两次“批量预测”，模型输出的结果大相径庭，模型第一次输出的结果是正常的，准确率非常高，第二次则是非常糟糕的胡乱预测。

经过各种测试，排除了 pyqt 原因。发现若调用两次 `infer()` 函数，所得值已经不同了。
```
if __name__ == '__main__':
    # 测试批量处理
    batch_predict_begin(compounds_files=r"E:\pyqt_bio_v3.0\test_data\2025-2-28_newsun\test.xlsx",save_model_path=r"E:\pyqt_bio_v3.0\tmp_Intermediate\best_model.pth")
    batch_predict_begin(compounds_files=r"E:\pyqt_bio_v3.0\test_data\2025-2-28_newsun\test.xlsx",save_model_path=r"E:\pyqt_bio_v3.0\tmp_Intermediate\best_model.pth")
```
结果分别是：
```
[tensor([0.4914, 0.5086]), tensor([0.7431, 0.2569]), tensor([0.7554, 0.2446]), tensor([0.7703, 0.2297]), tensor([0.4292, 0.5708]), tensor([0.4980, 0.5020]), tensor([0.4200, 0.5800]), tensor([0.5136, 0.4864]), tensor([0.4726, 0.5274])]
['活性', '非活性', '非活性', '非活性', '活性', '活性', '活性', '非活性', '活性']
============================================================================================================
[tensor([0.5284, 0.4716]), tensor([0.7869, 0.2131]), tensor([0.8038, 0.1962]), tensor([0.8193, 0.1807]), tensor([0.4723, 0.5277]), tensor([0.5348, 0.4652]), tensor([0.4633, 0.5367]), tensor([0.5571, 0.4429]), tensor([0.5085, 0.4915])]
['非活性', '非活性', '非活性', '非活性', '活性', '非活性', '活性', '非活性', '非活性']
```
原先未发现此类问题，因为无程序界面时，代码运行完预测函数后就会退出，不可能发生预测两次的情况。

更奇怪的是，若把 `infer()` 中的内容复制两份写到主函数里，效果等同于调用两次 `infer()` ，但却不会出现结果不同的情况！！

且已确认并不是预训练模型的问题，初步判断是三维特征加载与一二维特征加载时，顺序被打乱了，分子 A 的三维特征与分子 B 的一二维特征拼接了。

最终解决办法是，在 `infer()` 的开头固定随机种子，此后问题消失。但是已经仔细看过代码，自认为不存在随机加载的情况，不知道是哪一步具有随机性导致的先前问题。
```python
torch.manual_seed(42)
paddle.seed(42)
np.random.seed(42)
```
# 药物分子活性预测 PYQT 程序 V2.0 使用手册
## Start
### 从代码中运行 pyqt 方法
```
python Main.py
```
### 无需 pyqt ，仅训练与测试模型效果
```
python train.py
```
## Guide
1. (1)用于训练的分子文件必须包含 2 列： `canonical_smiles` （存储SMILES）和 `standard_value` （存储IC50值）;(2)需要预测活性的分子文件必须包含 1 列： `smiles` 。若您的文件列名不是这两个，有劳重命名一下（后续版本拟打算添加自动识别列名功能）。
2. 如果 IC50 的阈值是： 0-50 为活性，大于 100 为非活性，那么 50 到 100 之间的分子会被舍弃，仅保留小于 50 和大于 100 的分子。
3. 在选择阈值时，忽略 IC50 的单位 μMol ，直接写数字就好，不必进行单位转换。
4. 只能选择预处理后的文件存储目录，文件默认命名为： `//Preprocessed_compounds.xlsx` 。（后续拟增加自定义命名）。
5. 本程序自带了用于训练、测试的分子数据，位于 `test_data` 中。若手头没有分子数据，可用它测试程序。
6. 模型训练部分的三个参数设置了默认值，若无特殊需要，使用默认的即可。
7. 如果训练好了模型 A ，后续需要重新训练一个模型，最好选一个新的存储路径，为了防止原先训练的模型 A 被覆盖。目前存储模型的名字也无法自定义，后续拟增加该功能。

## Updata (相比于 V1.0 的新加功能)
- [x] 损失函数的权重因子自动根据数据集的正负样本比例计算，无需手动调整。计算方法：`ALPHA = 阳性样本/总样本`。
- [x] 增加了批量处理样本的功能。
- [x] 原先版本的数据预处理中不存在异常分子去除的功能，这些分子无法被提取二维结构，最终导致一维、二维样本个数不一致。本版本中新增了该功能。
- [x] 不设置固定不变的数字作为二维提取中的最大原子数，而是去寻找数据集中的分子最大原子数是多少，存储这个值，用作初始化节点矩阵的大小 `zero(max_atoms)` 【待测试是否会影响性能】
- [x] 增加了 logo  
- [x] 打包为 exe
## Todo (未来拟增加功能)
- [ ] 对存储分子三维信息的 SDF 文件的读取支持。目前仅能读取一维 SMILES 文件，需要通过处理提取转换出三维特征，若能够支持 SDF 则可跳过维度转换这一步，极大提高处理效率。
- [ ] 提供默认的测试数据，一起和 exe 程度打包，选择待处理分子文件之前，文本框显示默认测试数据的路径，以供用户在手头无数据的情况下测试程序。(低优先级)
- [ ] 自动识别分子文件的列名，避免用户重命名
- [ ] 增加 3D 处理部分会使得训练过程非常缓慢，在程序界面增加一个字段让用户选择是否使用 3D 功能
- [ ] 用户自主命名存储模型的名字是什么，而不是默认为 `best_model.pth`
- [ ] 添加对 cuda 的支持
- [ ] 对三维提取的测试
- [ ] 补全 `requirements.txt` 中所需库，或者换一种环境管理方式（如 PDM ）
## 记录
1. 高内聚、低耦合的方法
原本把 `train` 函数放到 UI 类里面，是因为该函数需要调用 UI 的 `text` 显示文字。但是最好把 `train` 独立放在一个新文件中，此时由于它是类中的函数， `train` 中本来需要传入 `self` 的，独立之后 `self` 没办法传入，而 train 又会被类内函数调用。

解决办法：把 `train` 的 `self` 参数改为 `ui` ，类内函数传入 `self.ui`。
## DirTree (项目目录与文件描述)
```
pyqt_bioactivity/
    ├── core_code/
    │   ├── d2_utils/
    │   │   ├── graph_embedding.py
    │   │   ├── node_embedding.py
    │   │   └── tools.py
    │   ├── d2_main.py
    │   ├── d3_infer.py
    │   ├── d3_main.py
    │   ├── dataset.py
    │   ├── featurizer.py
    │   ├── Focal_Loss.py
    │   └── MYmodel_final.py
    ├── Dependencies/
    │   ├── class.pdparams
    │   ├── geognn2.json
    │   └── model_300dim.pkl
    ├── log/
    │   └── 1.log
    ├── pahelix/
    │   ├── datasets/
    │   │   ├── __init__.py
    │   │   ├── bace_dataset.py
    │   │   ├── bbbp_dataset.py
    │   │   ├── chembl_filtered_dataset.py
    │   │   ├── clintox_dataset.py
    │   │   ├── davis_dataset.py
    │   │   ├── ddi_dataset.py
    │   │   ├── dti_dataset.py
    │   │   ├── esol_dataset.py
    │   │   ├── freesolv_dataset.py
    │   │   ├── hiv_dataset.py
    │   │   ├── inmemory_dataset.py
    │   │   ├── kiba_dataset.py
    │   │   ├── lipophilicity_dataset.py
    │   │   ├── mutag_dataset.py
    │   │   ├── muv_dataset.py
    │   │   ├── ogbg_molhiv_dataset.py
    │   │   ├── ogbg_molpcba_dataset.py
    │   │   ├── pdbbind_dataset.py
    │   │   ├── ppi_dataset.py
    │   │   ├── ptc_mr_dataset.py
    │   │   ├── qm7_dataset.py
    │   │   ├── qm8_dataset.py
    │   │   ├── qm9_dataset.py
    │   │   ├── qm9_gdb_dataset.py
    │   │   ├── sider_dataset.py
    │   │   ├── tox21_dataset.py
    │   │   ├── toxcast_dataset.py
    │   │   └── zinc_dataset.py
    │   ├── featurizers/
    │   │   ├── __init__.py
    │   │   ├── gem_featurizer.py
    │   │   ├── het_gnn_featurizer.py
    │   │   ├── lite_gem_featurizer.py
    │   │   └── pretrain_gnn_featurizer.py
    │   ├── model_zoo/
    │   │   ├── __init__.py
    │   │   ├── gem_model.py
    │   │   ├── light_gem_model.py
    │   │   ├── pretrain_gnns_model.py
    │   │   ├── protein_sequence_model.py
    │   │   ├── sd_vae_model.py
    │   │   └── seq_vae_model.py
    │   ├── networks/
    │   │   ├── __init__.py
    │   │   ├── basic_block.py
    │   │   ├── compound_encoder.py
    │   │   ├── gnn_block.py
    │   │   ├── involution_block.py
    │   │   ├── lstm_block.py
    │   │   ├── optimizer.py
    │   │   ├── pre_post_process.py
    │   │   ├── resnet_block.py
    │   │   └── transformer_block.py
    │   ├── tests/
    │   │   ├── __init__.py
    │   │   └── import_test.py
    │   ├── utils/
    │   │   ├── metrics/
    │   │   │   └── molecular_generation/
    │   │   │       ├── NP_Score/
    │   │   │       │   ├── __init__.py
    │   │   │       │   ├── npscorer.py
    │   │   │       │   ├── publicnp.model.gz
    │   │   │       │   └── README
    │   │   │       ├── SA_Score/
    │   │   │       │   ├── __init__.py
    │   │   │       │   ├── fpscores.pkl.gz
    │   │   │       │   ├── README
    │   │   │       │   └── sascorer.py
    │   │   │       ├── __init__.py
    │   │   │       ├── mcf.csv
    │   │   │       ├── metrics_.py
    │   │   │       ├── utils_.py
    │   │   │       └── wehi_pains.csv
    │   │   ├── tests/
    │   │   │   ├── data_utils_test.py
    │   │   │   └── splitters_test.py
    │   │   ├── __init__.py
    │   │   ├── basic_utils.py
    │   │   ├── compound_constants.py
    │   │   ├── compound_tools.py
    │   │   ├── data_utils.py
    │   │   ├── language_model_tools.py
    │   │   ├── protein_tools.py
    │   │   └── splitters.py
    │   ├── __init__.py
    │   └── cmdline.py
    ├── test_data/
    │   └── 2025-2-28_newsun/
    │       ├── test.xlsx
    │       ├── train.xlsx
    │       ├── ~$test.xlsx
    │       └── 测试集真实标签.xlsx
    ├── tmp/
    │   ├── 11best_model.pth
    │   ├── best_model.pth
    │   ├── CBoW_50dim.pt
    │   ├── fig1.jpg
    │   ├── fig2.jpg
    │   ├── graph_embedding_infer.npz
    │   ├── graph_embedding_single_infer.npz
    │   ├── infer_d2feature.npz
    │   ├── Preprocessed_compounds.xlsx
    │   ├── X_test_d1.pkl
    │   ├── X_test_d2.pkl
    │   ├── X_train_d1.pkl
    │   ├── X_train_d2.pkl
    │   ├── y_test.pkl
    │   └── y_train.pkl
    ├── ui/
    │   └── 基于多模态信息融合的药物分子活性预测系统v2.0.ui
    ├── .gitignore
    ├── batch_predict.py
    ├── dir_tree.py
    ├── logo.ico
    ├── logo.png
    ├── Main.py
    ├── readme.md
    ├── requirements.txt
    ├── train.py
    └── 基于多模态信息融合的药物分子活性预测系统v1.0说明书.docx
```
