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


import torch
import poptorch

import yaml
import fire
from pathlib import Path

from wenet.utils.init_model import init_model


class ExportPipeline:
    def __init__(
        self,
        ckpt_folder: str,
        ckpt_file: str,
        cmvn_file: str,
        vocab_file: str,
        config_file: str,
        output_folder: str,
        beam_size: int = 10,
        reverse_weight: float = -1.0,
        ctc_weight: float = -1.0,
    ):
        """
        ckpt_folder: folder for checkpoint
        ckpt_file: checkpoint file name
        cmvn_file: cmvn file name
        vocab_file: vocab file name
        config_file: config file name
        output_folder: output popef's folder
        """
        self.ckpt_folder = Path(ckpt_folder)
        self.ckpt_file = self.ckpt_folder.joinpath(ckpt_file)
        self.cmvn_file = self.ckpt_folder.joinpath(cmvn_file)
        self.vocab_file = self.ckpt_folder.joinpath(vocab_file)
        self.config_file = self.ckpt_folder.joinpath(config_file)
        self.output_folder = Path(output_folder)
        self.reverse_weight = reverse_weight
        self.beam_size = beam_size
        self.ctc_weight = ctc_weight

    def run(self):
        torch.manual_seed(0)
        torch.set_printoptions(precision=10)
        self._load_ckpt()
        self._compile_graph()
        self._check_precision()

    def _load_ckpt(self):
        assert self.ckpt_folder.exists()
        assert self.ckpt_file.exists()
        assert self.cmvn_file.exists()
        assert self.vocab_file.exists()
        assert self.config_file.exists()

        with open(self.config_file, 'r') as reader:
            configs = yaml.load(reader, Loader=yaml.FullLoader)
            configs['cmvn_file'] = str(self.cmvn_file)
            if self.reverse_weight != -1.0:
                configs['reverse_weight'] = self.reverse_weight
            if self.ctc_weight != -1.0:
                configs['ctc_weight'] = self.ctc_weight
            configs["encoder_conf"]["use_dynamic_chunk"] = False

        model = init_model(configs)
        model.load_state_dict(
            torch.load(self.ckpt_file, map_location='cpu'),
            strict=False)
        model.eval()
        print(model)

    def _compile_graph(self):
        pass

    def _check_precision(self):
        pass


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()


class StreamEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()


if __name__ == '__main__':
    fire.Fire(ExportPipeline)
