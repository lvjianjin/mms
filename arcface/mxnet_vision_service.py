# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
MXNetVisionService defines a MXNet base vision service
"""
import logging
import mxnet as mx
import numpy as np

from mxnet_model_service import MXNetModelService
from mxnet_utils import image, ndarray


class MXNetVisionService(MXNetModelService):
    """
    MXNetVisionService defines a fundamental service for image classification task.
    In preprocess, input image buffer is read to NDArray and resized respect to input
    shape in signature.
    In post process, top-5 labels are returned.
    """

    def preprocess(self, request):
        """
        Decode all input images into ndarray.

        Note: This implementation doesn't properly handle error cases in batch mode,
        If one of the input images is corrupted, all requests in the batch will fail.

        :param request:
        :return:
        """
        img_list = []
        param_name = self.signature['inputs'][0]['data_name']
        input_shape = self.signature['inputs'][0]['data_shape']

        for idx, data in enumerate(request):
            img = data['body'][param_name]

            if img is None or len(img) == 0:
                self.error = "Empty image input"
                return None

            # We are assuming input shape is NCHW
            [h, w] = input_shape[2:]

            try:
                # img_arr = image.read(img)
                img_arr = mx.nd.array(img)
            except Exception as e:
                logging.warn(e, exc_info=True)
                self.error = "Corrupted image input"
                return None

            img_arr = image.resize(img_arr, w, h)
            img_arr = image.transform_shape(img_arr)
            img_list.append(img_arr)
        return img_list

    def postprocess(self, data):
        emb = data[0].asnumpy()
        norm = np.sqrt(np.sum(emb * emb) + 0.00001)
        emb /= norm
        return emb.tolist()


_service = MXNetVisionService()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)
    if data is None:
        return None
    return _service.handle(data, context)
