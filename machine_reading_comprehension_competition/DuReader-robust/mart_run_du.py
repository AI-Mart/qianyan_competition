# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The HuggingFace Inc. team.
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

import os
import random
import time
import json
import math

from functools import partial
import numpy as np
import paddle

from paddle.io import DataLoader
from args import parse_args
import paddlenlp as ppnlp
from paddlenlp.data import Pad, Stack, Tuple, Dict
from paddlenlp.transformers import BertForQuestionAnswering, BertTokenizer
from paddlenlp.transformers import ErnieForQuestionAnswering, ErnieTokenizer
from paddlenlp.transformers import ErnieGramForQuestionAnswering, ErnieGramTokenizer
from paddlenlp.transformers import RobertaForQuestionAnswering, RobertaTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics.squad import squad_evaluate, compute_prediction
from paddlenlp.datasets import load_dataset
from mart_utils import prepare_train_features, prepare_validation_features, evaluate
from functools import partial


MODEL_CLASSES = {
    "bert": (BertForQuestionAnswering, BertTokenizer),
    "ernie": (ErnieForQuestionAnswering, ErnieTokenizer),
    "ernie_gram": (ErnieGramForQuestionAnswering, ErnieGramTokenizer),
    "roberta": (RobertaForQuestionAnswering, RobertaTokenizer)
}


def set_seed(args_):
    """
    pass
    """
    random.seed(args_.seed)
    np.random.seed(args_.seed)
    paddle.seed(args_.seed)


class CrossEntropyLossForSQuAD(paddle.nn.Layer):
    """
    pass
    """
    def __init__(self):
        super(CrossEntropyLossForSQuAD, self).__init__()

    def forward(self, y, label):
        """
        pass
        """
        start_logits, end_logits = y
        start_position, end_position = label
        start_position = paddle.unsqueeze(start_position, axis=-1)
        end_position = paddle.unsqueeze(end_position, axis=-1)
        start_loss = paddle.nn.functional.cross_entropy(
            input=start_logits, label=start_position)
        end_loss = paddle.nn.functional.cross_entropy(
            input=end_logits, label=end_position)
        loss = (start_loss + end_loss) / 2
        return loss


def run(args_):
    """
    pass
    """
    paddle.set_device(args_.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    rank = paddle.distributed.get_rank()

    task_name = args_.task_name.lower()
    train_ds, dev_ds, test_ds = load_dataset(task_name, splits=('train', 'dev', 'test'))
    model_type = args_.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[model_type]
    tokenizer = tokenizer_class.from_pretrained(args_.model_name_or_path)
    set_seed(args_)
    if rank == 0:
        if os.path.exists(args_.model_name_or_path):
            print("init checkpoint from %s" % args_.model_name_or_path)

    model = model_class.from_pretrained(args_.model_name_or_path)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    train_trans_func = partial(prepare_train_features,
                               max_seq_length=args_.max_seq_length,
                               doc_stride=args_.doc_stride,
                               tokenizer=tokenizer)

    train_ds.map(train_trans_func, batched=True, num_workers=4)

    dev_trans_func = partial(prepare_validation_features,
                             max_seq_length=args_.max_seq_length,
                             doc_stride=args_.doc_stride,
                             tokenizer=tokenizer)

    dev_ds.map(dev_trans_func, batched=True, num_workers=4)
    test_ds.map(dev_trans_func, batched=True, num_workers=4)

    # 定义BatchSampler
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args_.batch_size, shuffle=True)

    dev_batch_sampler = paddle.io.BatchSampler(
        dev_ds, batch_size=args_.batch_size, shuffle=False)

    test_batch_sampler = paddle.io.BatchSampler(
        test_ds, batch_size=args_.batch_size, shuffle=False)

    # 定义batchify_fn
    train_batchify_fn = lambda samples, fn=Dict({
        "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
        "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        "start_positions": Stack(dtype="int64"),
        "end_positions": Stack(dtype="int64")
    }): fn(samples)

    dev_batchify_fn = lambda samples, fn=Dict({
        "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
        "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id)
    }): fn(samples)

    # 构造DataLoader
    train_data_loader = paddle.io.DataLoader(
        dataset=train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=train_batchify_fn,
        return_list=True)

    dev_data_loader = paddle.io.DataLoader(
        dataset=dev_ds,
        batch_sampler=dev_batch_sampler,
        collate_fn=dev_batchify_fn,
        return_list=True)

    test_data_loader = paddle.io.DataLoader(
        dataset=test_ds,
        batch_sampler=test_batch_sampler,
        collate_fn=dev_batchify_fn,
        return_list=True)

    num_training_steps = args_.max_steps if args_.max_steps > 0 else len(
        train_data_loader) * args_.num_train_epochs
    num_train_epochs = math.ceil(num_training_steps /
                                 len(train_data_loader))

    lr_scheduler = LinearDecayWithWarmup(
        args_.learning_rate, num_training_steps, args_.warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=args_.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args_.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)
    criterion = CrossEntropyLossForSQuAD()
    if args_.do_train:
        global_step = 0
        tic_train = time.time()
        for epoch in range(num_train_epochs):
            for step, batch in enumerate(train_data_loader):
                global_step += 1
                input_ids, token_type_ids, start_positions, end_positions = batch
                logits = model(
                    input_ids=input_ids, token_type_ids=token_type_ids)
                loss = criterion(logits, (start_positions, end_positions))

                if global_step % args_.logging_steps == 0:
                    print(
                        "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                        % (global_step, epoch + 1, step + 1, loss,
                           args_.logging_steps / (time.time() - tic_train)))
                    tic_train = time.time()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()

                if global_step % args_.save_steps == 0 or global_step == num_training_steps:
                    if rank == 0:
                        evaluate(model=model, data_loader=dev_data_loader, args=args_)
                        output_dir = os.path.join(args_.output_dir,
                                                  "model_%d" % global_step)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # need better way to get inner model of DataParallel
                        model_to_save = model._layers if isinstance(
                            model, paddle.DataParallel) else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        print('Saving checkpoint to:', output_dir)
                    if global_step == num_training_steps:
                        break

        evaluate(model=model, data_loader=test_data_loader, args=args_, is_test=True)
# 可以直接在这里评估，不必加载，加载是因为经过前面是初始化了，需要重新加载训练好的，这里一直在for执行，结束的参数就是最终的参数
    else:
        evaluate(model=model, data_loader=test_data_loader, args=args_, is_test=True)


if __name__ == "__main__":
    args = parse_args()
    run(args)
