# Copyright (c) 2022 Graphcore Ltd. 
#               2022 JinLe Tong (richardt@graphcore.ai)
# All rights reserved.
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

import argparse
import torch
import poptorch



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='export ipu model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--cmvn_file', required=False, default='', type=str,
                        help='global_cmvn file, default path is in config file')
    parser.add_argument('--reverse_weight', default=-1.0, type=float,
                        required=False,
                        help='reverse weight for bitransformer,' +
                        'default value is in config file')
    parser.add_argument('--ctc_weight', default=-1.0, type=float,
                        required=False,
                        help='ctc weight, default value is in config file')
    parser.add_argument('--beam_size', default=10, type=int, required=False,
                        help="beam size would be ctc output size")
    parser.add_argument('--output_onnx_dir',
                        default="onnx_model",
                        help='output onnx encoder and decoder directory')
    parser.add_argument('--fp16',
                        action='store_true',
                        help='whether to export fp16 model, default false')
    # arguments for streaming encoder
    parser.add_argument('--streaming',
                        action='store_true',
                        help="whether to export streaming encoder, default false")
    parser.add_argument('--decoding_chunk_size',
                        default=16,
                        type=int,
                        required=False,
                        help='the decoding chunk size, <=0 is not supported')
    parser.add_argument('--num_decoding_left_chunks',
                        default=5,
                        type=int,
                        required=False,
                        help="number of left chunks, <= 0 is not supported")
    args = parser.parse_args()