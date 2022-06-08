import tvm
from tvm.meta_schedule import tune_tir
from tvm.meta_schedule import TuneConfig, tune_tir
from tvm.script import tir as T
from tvm.target import Target

import tempfile


@T.prim_func
def Dense(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [2048, 768])
    B = T.match_buffer(b, [768, 2304])
    C = T.match_buffer(c, [2048, 2304])
    for i, j, k in T.grid(2048, 2304, 768):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def test_dense_cuda():
    tir_sched = tune_tir(
                    mod=Dense,
                    target=Target("nvidia/geforce-rtx-3090"),
                    config=TuneConfig(
                        strategy="replay_trace",
                        num_trials_per_iter=32,
                        max_trials_per_task=32,
                        max_trials_global=32,
                    ),
                    work_dir="./tuning_record",
                )
    if tir_sched is None:
        print("No valid schedule found!")
    else:
        print(tir_sched.mod.script())
        print(tir_sched.trace)
        print(tvm.lower(tir_sched.mod["main"], []))
