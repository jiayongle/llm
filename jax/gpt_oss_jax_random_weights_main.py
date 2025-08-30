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

# install gpt_jax_oss in https://github.com/jax-ml/jax-llm-examples/tree/main/gpt_oss

import dataclasses
from etils import epath
import json
from pprint import pprint
from functools import partial

import jax
from jax import numpy as jnp
from jax import random
from jax.sharding import set_mesh, AxisType, PartitionSpec as P

try:
    from jax.sharding import use_mesh as set_mesh  # jax < 0.7.0
except ImportError:
    pass
import numpy as np

from gpt_oss_jax import model as gpt_jax


jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax_cache").expanduser()))


# replacing top-k might be necessary for XLA:GPU
@partial(jax.jit, static_argnums=(1,))
def top_k(op, k):
    idxs = jnp.argsort(op, descending=True, axis=-1)[..., :k]
    return jnp.take_along_axis(op, idxs, axis=-1), idxs

jax.lax.top_k = top_k


def encode_input(tokenizer, texts, pad_id: int = gpt_jax.PAD_ID):
    if tokenizer is None:
      # return jnp.ones((len(texts), 64), jnp.int32)
      return random.randint(random.key(0), (len(texts), 64), 0, 10_000, dtype=jnp.int32)
    assert isinstance(texts, list)
    inputs = [
        tokenizer.apply_chat_template([{"role": "user", "content": text}], add_bos=True, add_generation_prompt=True)
        for text in texts
    ]
    max_len = max([len(x) for x in inputs])
    inputs = [(max_len - len(x)) * [pad_id] + x for x in inputs]
    return np.array(inputs)

CONFIG_20B = gpt_jax.Config(embed=2880,
       q_heads=64,
       kv_heads=8,
       num_layers=24,
       head_dim=64,
       vocab_size=201088,
       max_seq_len=2048,
       causal=True,
       sliding_attention_map=['sliding_attention',
                              'full_attention',
                              'sliding_attention',
                              'full_attention',
                              'sliding_attention',
                              'full_attention',
                              'sliding_attention',
                              'full_attention',
                              'sliding_attention',
                              'full_attention',
                              'sliding_attention',
                              'full_attention',
                              'sliding_attention',
                              'full_attention',
                              'sliding_attention',
                              'full_attention',
                              'sliding_attention',
                              'full_attention',
                              'sliding_attention',
                              'full_attention',
                              'sliding_attention',
                              'full_attention',
                              'sliding_attention',
                              'full_attention'],
       sliding_window_size=128,
       moe_ffw_size=2880,
       moe_experts_per_tok=4,
       moe_num_experts=32,
    )

CONFIG_120B = gpt_jax.Config(embed=2880,
       q_heads=64,
       kv_heads=8,
       num_layers=36,
       head_dim=64,
       vocab_size=201088,
       max_seq_len=2048,
       causal=True,
       sliding_attention_map=['sliding_attention',
                              'full_attention',
                              'sliding_attention',
                              'full_attention',
                              'sliding_attention',
                              'full_attention',
                              'sliding_attention',
                              'full_attention',
                              'sliding_attention',
                              'full_attention',
                              'sliding_attention',
                              'full_attention',
                              'sliding_attention',
                              'full_attention',
                              'sliding_attention',
                              'full_attention',
                              'sliding_attention',
                              'full_attention',
                              'sliding_attention',
                              'full_attention',
                              'sliding_attention',
                              'full_attention',
                              'sliding_attention',
                              'full_attention',
                              'sliding_attention',
                              'full_attention',
                              'sliding_attention',
                              'full_attention',
                              'sliding_attention',
                              'full_attention',
                              'sliding_attention',
                              'full_attention',
                              'sliding_attention',
                              'full_attention',
                              'sliding_attention',
                              'full_attention'],
       sliding_window_size=128,
       moe_ffw_size=2880,
       moe_experts_per_tok=4,
       moe_num_experts=128,
)



if __name__ == "__main__":
    # jax.distributed.initialize()  # if you want to run multi-host
    quant = True

    #ckpt_path = epath.Path("~/bucket/gpt_oss_jax/gpt_oss_20b").expanduser()
    ckpt_path = epath.Path("~/bucket/gpt_oss_jax/gpt_oss_120b").expanduser()
    if quant:
        ckpt_path = ckpt_path.parent / f"{ckpt_path.name}-quant"
    # from transformers import AutoTokenizer
    #tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    tokenizer = None

    tp = 2
    mesh = jax.make_mesh(
        (1, tp, jax.device_count() // tp), ("x", "y", "z"), devices=jax.devices(), axis_types=(AxisType.Explicit,) * 3
    )

    # cfg = gpt_jax.hf_to_jax_config(json.loads((ckpt_path / "config.json").read_text()))
    # cfg = CONFIG_120B
    cfg = CONFIG_20B

    cfg = dataclasses.replace(cfg, mesh=mesh, quant_moe=quant, quant_cache=quant, max_seq_len=2048)

    pprint(cfg)

    input = encode_input(
        tokenizer,
        [
            "Tell me your name",
            "What is the weather like expressed in long prose in Old English",
            "Do you like ice cream, be extremely precise",
        ]
        + ["Do you like ice cream, be extremely precise"] * (4 - 3),
    )
    # kweights = gpt_jax.load_pytree(ckpt_path, gpt_jax.Weights.shardings(cfg))
    weights = gpt_jax.Weights.init(random.key(0), cfg)
    weights = jax.device_put(weights, gpt_jax.compute_optimal_weights_layouts(weights, cfg))

    profile = True
    with set_mesh(cfg.mesh):
        zero_cache = gpt_jax.KVCache.init(random.key(1), cfg, input.shape[0], cfg.max_seq_len)
        next_tokens, logits, cache = gpt_jax.prefill(input, weights, zero_cache, cfg)
        curr_tokens = next_tokens.at[:, cache.iter - 1 : cache.iter].get(out_sharding=P(None, None))
        tokens_list = []
        for i in range(1024):
            if profile and i == 2:
                jax.profiler.start_trace("/tmp/gpt_profile")
            tokens_list.append(curr_tokens)
            curr_tokens, cache = gpt_jax.decode_step(curr_tokens, weights, cache, cfg)
            if profile and i == 6:
                jax.block_until_ready(tokens_list)
                jax.profiler.stop_trace()
        tokens = np.array(jnp.concatenate(tokens_list, axis=-1))
    if tokenizer is not None:
        responses = [tokenizer.decode(row) for row in tokens]
    else:
        responses = tokens
    print("Responses:")
    for response in responses:
        print(response)
        print("\n".join(3 * ["-" * 80]))
