# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from etils import epath
from pprint import pformat

import jax
from jax import numpy as jnp
from jax import random
import numpy as np

from deepseek_r1_jax import model as dsjax
from deepseek_r1_jax import chkpt_utils as utils


def encode_input(tokenizer, texts, pad_id: int = dsjax.PAD_ID):
    assert isinstance(texts, list)
    inputs = [
        tokenizer.apply_chat_template([{"role": "user", "content": text}]) + tokenizer.encode("<|Assistant|><think>")
        for text in texts
    ]
    max_len = max([len(x) for x in inputs])
    inputs = [(max_len - len(x)) * [pad_id] + x for x in inputs]
    return np.array(inputs)


if __name__ == "__main__":
    # jax.distributed.initialize()  # skip on single node
    #ckpt_path = epath.Path(f"~/bucket/deepseek-r1-jax-chkpt").expanduser()
    #tokenizer = dsjax.load_tokenizer(ckpt_path / "tokenizer.json", ckpt_path / "tokenizer_config.json")
    tokenizer = dsjax.load_tokenizer()

    input = encode_input(
        tokenizer,
        [
            "Tell me your name",
            "What is the weather like expressed in long prose in Old English",
            "Do you like ice cream, be extremely precise",
        ],
    )

    mesh = jax.make_mesh((1, 4, jax.device_count() // 4), ("x", "y", "z"), devices=jax.devices())
    cfg = dataclasses.replace(dsjax.Config(), mesh=mesh)
    cfg.use_prefill_attn_kernel = False
    cfg.use_decode_attn_kernel = False
    cfg.use_decode_ragged_dot_kernel = False

    #weights = utils.load_model(epath.Path(ckpt_path).expanduser(), cfg)
    weights = dsjax.Weights.init(random.key(0), cfg)

    zero_cache = dsjax.KVCache.init(random.key(1), cfg, input.shape[0], cfg.max_seq_len)
    curr_tokens, logits, cache = dsjax.prefill(input, weights, zero_cache, cfg)
    curr_tokens, tokens_list = curr_tokens[:, cache.iter - 1 : cache.iter], []
    tokens_list = []
    for i in range(32):
        if i == 2:
            jax.profiler.start_trace("/tmp/my_profile")
        tokens_list.append(curr_tokens)
        curr_tokens, cache = dsjax.decode_step(curr_tokens, weights, cache, cfg)
        if i == 31:
            jax.block_until_ready(tokens_list)
            jax.profiler.stop_trace()
    tokens = np.array(jnp.concatenate(tokens_list, axis=-1))
    responses = [tokenizer.decode(row) for row in tokens]
    print("Responses:\n" + pformat(responses))
