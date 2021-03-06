#!/usr/bin/env python3
# -*- coding:utf-8 -*-
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

"""main interface of text2sql model

Filname: text2sql_main.py
Authors: ZhangAo(@baidu.com)
Date: 2020-10-14 14:08:10
"""

import sys
import os
import traceback
import logging
import json
from pathlib import Path
from functools import partial
import random

import numpy as np
import paddle
import paddle.distributed as dist
# 在这里 import 只是为了避免使用 TSL 库的 bug
# 详见 https://github.com/pytorch/pytorch/issues/2575#issuecomment-523657178
from paddlenlp.transformers import BertTokenizer

import text2sql
from text2sql import global_config
from text2sql import dataproc
from text2sql import launch
from text2sql.grammars.dusql_v2 import DuSQLLanguageV2
from text2sql.grammars.nl2sql import NL2SQLLanguage


ModelClass = None
GrammarClass = None
DataLoaderClass = None
DatasetClass = None
g_input_encoder = None
g_label_encoder = None


def preprocess(config):
    """pre-process datasets and rules

    Args:
        config (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    dataset_config = {
            'db_file': config.data.db,
            'input_encoder': g_input_encoder,
            'label_encoder': g_label_encoder,
            'is_cached': False,}

    output_base = config.data.output
    if config.data.train_set is not None:
        dataset = DatasetClass(name='train', data_file=config.data.train_set, **dataset_config)
        dataset.save(output_base, save_db=True)
        # label encoder 只能“看到”训练集
        g_label_encoder.save(Path(output_base) / 'label_vocabs')

    if config.data.dev_set is not None:
        dataset = DatasetClass(name='dev', data_file=config.data.dev_set, **dataset_config)
        dataset.save(output_base, save_db=False)

    if config.data.test_set is not None:
        dataset = DatasetClass(name='test', data_file=config.data.test_set, **dataset_config)
        dataset.save(output_base, save_db=False)


def train(config):
    """do training

    Args:
        params (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    logging.info('training arguments: %s', config)
    if config.train.use_data_parallel:
        logging.info("parallel mode. init env...")
        dist.init_parallel_env()

    dataset_config = {'db_file': config.data.db,
                      'input_encoder': g_input_encoder,
                      'label_encoder': g_label_encoder,
                      'is_cached': True}
    train_set = DatasetClass(name='train', data_file=config.data.train_set, **dataset_config)
    dev_set = DatasetClass(name='dev', data_file=config.data.dev_set, **dataset_config)

    shuf_train = True if not config.general.is_debug else False
    train_reader = DataLoaderClass(config, train_set, batch_size=config.general.batch_size, shuffle=shuf_train)
    #dev_reader = dataproc.DataLoader(config, dev_set, batch_size=config.general.batch_size, shuffle=False)
    dev_reader = DataLoaderClass(config, dev_set, batch_size=1, shuffle=False)
    max_train_steps = config.train.epochs * (len(train_set) // config.general.batch_size // config.train.trainer_num)

    model = ModelClass(config.model, g_label_encoder)
    if config.model.init_model_params is not None:
        logging.info("loading model param from %s", config.model.init_model_params)
        model.set_state_dict(paddle.load(config.model.init_model_params))
    if config.train.use_data_parallel:
        logging.info("parallel mode. init model...")
        #strategy = paddle.distributed.prepare_context()
        #model = paddle.DataParallel(model, strategy)
        model = paddle.DataParallel(model)

    optimizer = text2sql.optim.init_optimizer(model, config.train, max_train_steps)
                                              #scale_params_lr=[(model.encoder.ernie, 0.1)])
    if config.model.init_model_optim is not None:
        logging.info("loading model optim from %s", config.model.init_model_optim)
        optimizer.set_state_dict(paddle.load(config.model.init_model_optim))

    logging.info("start of training...")
    launch.trainer.train(config, model, optimizer, config.train.epochs, train_reader, dev_reader)
    logging.info("end of training...")


def inference(config):
    """

    Args:
        config (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    if config.model.init_model_params is None:
        raise RuntimeError("config.init_model_params should be a valid model path")

    dataset_config = {'db_file': config.data.db,
                      'input_encoder': g_input_encoder,
                      'label_encoder': g_label_encoder,
                      'is_cached': True}
    test_set = DatasetClass(name='test', data_file=config.data.test_set, **dataset_config)
    # batch_size 恒为 1
    test_reader = DataLoaderClass(config, test_set, batch_size=1, shuffle=False)

    model = ModelClass(config.model, g_label_encoder)
    logging.info("loading model param from %s", config.model.init_model_params)
    state_dict = paddle.load(config.model.init_model_params)
    model.set_state_dict(state_dict)

    logging.info("start of inference...")
    launch.infer.inference(model, test_reader, config.data.output,
                           beam_size=config.general.beam_size,
                           model_name=config.model.model_name)
    logging.info("end of inference...")


def evaluate(config):
    """alpha version

    Args:
        params (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    dataset_config = {'db_file': config.data.db,
                      'input_encoder': g_input_encoder,
                      'label_encoder': g_label_encoder,
                      'is_cached': True,
                      'schema_file': config.data.db_schema}
    test_set = DatasetClass(name='test', data_file=config.data.test_set, **dataset_config)
    with open(config.data.eval_file) as ifs:
        infer_results = list(ifs)
    # batch_size 恒为 1
    #test_reader = DataLoaderClass(config, test_set, batch_size=1, shuffle=False)

    model = None
    #model = text2sql.models.EncDecModel(g.device, None, g_label_encoder)
    #if config.init_model_params is None:
    #    raise RuntimeError("config.init_model_params should be a valid model path")
    #logging.info("loading model param from %s", config.init_model_params)
    #model.set_dict(paddle.load(config.init_model_params))

    logging.info("start of evaluating...")
    launch.eval.evaluate(model, test_set, infer_results, eval_value=config.general.is_eval_value)
    logging.info("end of evaluating....")


def init_env(config):
    """init system env

    Args:
        config (obj): NULL

    Returns: None

    Raises: NULL
    """
    log_level = logging.INFO if not config.general.is_debug else logging.DEBUG
    formater = logging.Formatter('%(levelname)s %(asctime)s %(filename)s:%(lineno)03d * %(message)s')
    logger = logging.getLogger()
    logger.setLevel(log_level)
    handler = logger.handlers[0]
    handler.setLevel(log_level)
    handler.setFormatter(formater)

    seed = config.train.random_seed
    if seed is not None:
        random.seed(seed)
        paddle.seed(seed)
        np.random.seed(seed)

    global ModelClass
    global GrammarClass
    global DatasetClass
    global DataLoaderClass
    global g_input_encoder
    global g_label_encoder

    if config.model.grammar_type == 'dusql_v2':
        GrammarClass = DuSQLLanguageV2
    elif config.model.grammar_type == 'nl2sql':
        GrammarClass = NL2SQLLanguage
    else:
        raise ValueError('grammar type is not supported: %s' % (config.model.grammar_type))
    g_label_encoder = dataproc.SQLPreproc(config.data.grammar,
                                   GrammarClass,
                                   predict_value=config.model.predict_value,
                                   is_cached=config.general.mode != 'preproc')

    assert config.model.model_name == 'seq2tree_v2', 'only seq2tree_v2 is supported'
    g_input_encoder = dataproc.ErnieInputEncoderV2(config.model)
    ModelClass = lambda x1, x2: text2sql.models.EncDecModel(x1, x2, 'v2')
    DatasetClass = dataproc.DuSQLDatasetV2
    DataLoaderClass = partial(dataproc.DataLoader, collate_fn=dataproc.dataloader.collate_batch_data_v2)


def _set_proc_name(config, tag_base):
    """set process name on local machine
    """
    if config.general.is_cloud:
        return
    if tag_base.startswith('train'):
        tag_base = 'train'
    import setproctitle
    setproctitle.setproctitle(tag_base + '_' + config.data.output.rstrip('/').split('/')[-1])


if __name__ == "__main__":
    # 从命令行和配置文件中生成全局配置
    config = global_config.gen_config()
    init_env(config)

    run_mode = config.general.mode
    if run_mode == 'preproc':
        preprocess(config)
        sys.exit(0)

    _set_proc_name(config, run_mode)
    if run_mode == 'test':
        evaluate(config)
    elif run_mode == 'infer':
        inference(config)
    elif run_mode.startswith('train'):
        if config.train.use_data_parallel:
            dist.spawn(train, args=(config, ))
        else:
            train(config)

