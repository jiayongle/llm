import jax
from jax import numpy as jnp

@jax.named_call
def foo(x, y):
    return (x + y) / 2.

@jax.jit
def bar(a):
    def foo2(x, y):
        return foo(x, y), None

    out, _ = jax.lax.scan(foo2, 0., a)
    return out

a = jnp.array([1., 2., 3., 4., 5.])

print(jax.devices())
jax.profiler.start_trace('/tmp/tensorboard')
with jax.profiler.StepTraceAnnotation('step', step_num=0): # JIT warm-up
    out = bar(a)
with jax.profiler.StepTraceAnnotation('step', step_num=1):
    out = bar(a)
out.block_until_ready()
jax.profiler.stop_trace()
