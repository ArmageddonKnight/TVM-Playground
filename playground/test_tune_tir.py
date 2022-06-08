import tvm
from tvm.meta_schedule import tune_tir, TuneConfig
from tvm.meta_schedule.tune import Parse
from tvm.script import tir as T
from tvm.tir import Schedule
from tvm.target import Target


@T.prim_func
def Dense_2048x768x2304(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [2048, 768])
    B = T.match_buffer(b, [768, 2304])
    C = T.match_buffer(c, [2048, 2304])
    for i, j, k in T.grid(2048, 2304, 768):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@T.prim_func
def Dense_1920x768x2304(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [1920, 768])
    B = T.match_buffer(b, [768, 2304])
    C = T.match_buffer(c, [1920, 2304])
    for i, j, k in T.grid(1920, 2304, 768):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def test_dense_cuda_train():
    tir_sched = tune_tir(
                    mod=Dense_2048x768x2304,
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


def get_tuning_outcome(work_dir):
    database = Parse._database(None, work_dir)
    mod = Parse._mod(Dense_2048x768x2304)
    bests = database.get_top_k(
                database.commit_workload(mod),
                top_k=1,
            )
    sched = Schedule(mod)
    bests[0].trace.apply_to_schedule(sched, remove_postproc=False)
    return sched


def test_dense_cuda_infer():
    tir_sched = get_tuning_outcome('./tuning_record')

    if tir_sched is None:
        print("No valid schedule found!")
    else:
        print(tir_sched.mod.script())
        with open('./tuning_record/sample_sched.py', 'w') as fout:
            fout.write("""\
def Dense_2048x768x2304(sch):
    {}
""".format(str(tir_sched.trace).replace('\n', '\n    ')))
        print(tvm.lower(tir_sched.mod["main"], []))
