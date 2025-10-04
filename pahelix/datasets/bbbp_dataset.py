#!/usr/bin/python
#-*-coding:utf-8-*-
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Processing of Blood-Brain Barrier Penetration dataset

The Blood-brain barrier penetration (BBBP) dataset is extracted from a study on the modeling and 
prediction of the barrier permeability. As a membrane separating circulating blood and brain extracellular fluid, the blood-brain barrier blocks most drugs, hormones and neurotransmitters. Thus penetration of the barrier forms a long-standing issue in development of drugs targeting central nervous system.
This dataset includes binary labels for over 2000 compounds on their permeability properties.

You can download the dataset from
http://moleculenet.ai/datasets-1 and load it into pahelix reader creators
"""

import os
from os.path import join, exists
import pandas as pd
import numpy as np

from pahelix.datasets.inmemory_dataset import InMemoryDataset

#这个函数的作用是

def load_bioactivity_dataset(data_path):
    """
    data_path：输入要求是excel文件，其中包含两列smiles和label
    Returns:
        an InMemoryDataset instance.
    """


    input_df=pd.read_excel(data_path)
    smiles_list = input_df['smiles']
    from rdkit.Chem import AllChem
    #把smiles转换为mol
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    #把mol列表中为空的行删去
    preprocessed_rdkit_mol_objs_list = [m if not m is None else None for m in
                                                          rdkit_mol_objs_list]
    #再把mol转换 回smiles  ???
    smiles_list = [AllChem.MolToSmiles(m) if not m is None else
                                None for m in preprocessed_rdkit_mol_objs_list]
    labels = input_df['label']
    # convert 0 to -1。使用-1来代替0。此后的负标签是-1了
    labels = labels.replace(0, -1)
    # there are no nans

    data_list = []
    for i in range(len(smiles_list)):
        if smiles_list[i] is None:
            continue
        data = {}
        data['smiles'] = smiles_list[i]        
        data['label'] = labels.values[i]
        data_list.append(data)
    dataset = InMemoryDataset(data_list)
    return dataset
