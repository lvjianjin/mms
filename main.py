# -*- coding: utf-8 -*-
# @Time: 2021/1/20 17:27
# @Author: JianjinL
# @eMail: jianjinlv@163.com
# @File: main
# Software: PyCharm

import mxnet as mx

model = 'arcface1'
if model == 'arcface':
    model_path = "./arcface/model"
    input_shape = (1, 3, 112, 112)
else:
    model_path = "./retinaface/mnet12"
    input_shape = (1, 3, 112, 112)
# 加载模型
load_symbol, args, auxs = mx.model.load_checkpoint(model_path, 0)
mod = mx.mod.Module(load_symbol, label_names=None, data_names=['data'], context=mx.cpu())
mod.bind(data_shapes=[('data', input_shape)])
print(mod.data_names)
print(mod.data_shapes)
print(mod.output_names)
print(mod.output_shapes)