import tvm
from tvm.meta_schedule import tune_tir, TuneConfig, default_config
from tvm.script import tir as T
from tvm.tir import Schedule
from tvm.tir.transform import *
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
def Dense_960x770x2304(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [960, 770])
    B = T.match_buffer(b, [770, 2304])
    C = T.match_buffer(c, [960, 2304])
    for i, j, k in T.grid(960, 2304, 770):
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
    database = default_config.database(None, work_dir)
    mod = default_config.mod(Dense_2048x768x2304)
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
        with open('sample_sched.py', 'w') as fout:
            fout.write("""\
def Dense_2048x768x2304_sample_sched(sch):
    {}
""".format(str(tir_sched.trace).replace('\n', '\n    ')))
        print(tvm.lower(tir_sched.mod["main"], []))


def preprocess(mod):
    """
    Pre-process the IRModule by lowering the BlockScope's.
    """
    mod = LowerInitBlock()(mod)
    mod = PlanAndUpdateBufferAllocationLocation()(mod)
    mod = ConvertBlocksToOpaque()(mod)
    mod = CompactBufferAllocation()(mod)
    mod = Simplify()(mod)
    # mod = FlattenBuffer()(mod)
    # mod = Simplify()(mod)
    return mod


def test_dense_cuda_sample_sched_infer():
    from sample_sched import Dense_2048x768x2304_sample_sched
    from tvm.tir.transform import LocalPad

    tir_sched = Schedule(Dense_960x770x2304)
    Dense_2048x768x2304_sample_sched(tir_sched)

    if tir_sched is None:
        print("No valid schedule found!")
    else:
        print(tir_sched.mod.script())
        print(preprocess(tir_sched.mod))
        # print(tvm.lower(tir_sched.mod, []))


@tvm.script.ir_module
class SampleModule:
    @T.prim_func
    def main(A: T.Buffer[(960, 770), "float32"], B: T.Buffer[(770, 2304), "float32"], C: T.Buffer[(960, 2304), "float32"]) -> None:
        # body
        # with T.block("root")
        C_local = T.alloc_buffer([960, 2304], dtype="float32", scope="local")
        A_shared = T.alloc_buffer([960, 770], dtype="float32", scope="shared")
        B_shared = T.alloc_buffer([770, 2304], dtype="float32", scope="shared")
        for i_0_j_0_fused in T.thread_binding(144, thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step":512, "pragma_unroll_explicit":1}):
            for i_1_j_1_fused in T.thread_binding(2, thread="vthread.x"):
                for i_2_j_2_fused in T.thread_binding(256, thread="threadIdx.x"):
                    for i_3_init, j_3_init, i_4_init, j_4_init in T.grid(8, 1, 2, 2):
                        with T.block("update_init"):
                            vi = T.axis.spatial(960, (((i_0_j_0_fused // 18 + 0) * 8 + i_2_j_2_fused // 32) * 8 + i_3_init) * 2 + i_4_init)
                            vj = T.axis.spatial(2304, ((i_0_j_0_fused % 18 * 2 + i_1_j_1_fused % 2) * 32 + i_2_j_2_fused % 32 + j_3_init) * 2 + j_4_init)
                            T.where((((i_0_j_0_fused // 18 + 0) * 8 + i_2_j_2_fused // 32) * 8 + i_3_init) * 2 + i_4_init < 960)
                            T.reads()
                            T.writes(C_local[vi, vj])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive":1024, "meta_schedule.thread_extent_low_inclusive":32, "meta_schedule.tiling_structure":"SSSRRSRS"})
                            C_local[vi, vj] = T.float32(0)
                    for k_0 in T.serial(193):
                        for ax0_ax1_fused_0 in T.serial(1):
                            for ax0_ax1_fused_1 in T.thread_binding(256, thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(3):
                                    with T.block("A_shared"):
                                        v0 = T.axis.spatial(960, i_0_j_0_fused // 18 * 128 + (ax0_ax1_fused_0 * 768 + ax0_ax1_fused_1 * 3 + ax0_ax1_fused_2) // 4)
                                        v1 = T.axis.spatial(770, k_0 * 4 + (ax0_ax1_fused_0 * 768 + ax0_ax1_fused_1 * 3 + ax0_ax1_fused_2) % 4)
                                        T.where(i_0_j_0_fused // 18 * 128 + ((ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1) * 3 + ax0_ax1_fused_2) // 4 < 960 and k_0 * 4 + ((ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1) * 3 + ax0_ax1_fused_2) % 4 < 770 and (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1) * 3 + ax0_ax1_fused_2 < 512)
                                        T.reads(A[v0, v1])
                                        T.writes(A_shared[v0, v1])
                                        A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_0 in T.serial(1):
                            for ax0_ax1_fused_1 in T.thread_binding(256, thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(4):
                                    with T.block("B_shared"):
                                        v0 = T.axis.spatial(770, k_0 * 4 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 4 + ax0_ax1_fused_2) // 128)
                                        v1 = T.axis.spatial(2304, i_0_j_0_fused % 18 * 128 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 4 + ax0_ax1_fused_2) % 128)
                                        T.where(k_0 * 4 + ((ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1) * 4 + ax0_ax1_fused_2) // 128 < 770 and (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1) * 4 + ax0_ax1_fused_2 < 512)
                                        T.reads(B[v0, v1])
                                        T.writes(B_shared[v0, v1])
                                        B_shared[v0, v1] = B[v0, v1]
                        for k_1, i_3, j_3, k_2, i_4, j_4 in T.grid(1, 8, 1, 4, 2, 2):
                            with T.block("update_update"):
                                vi = T.axis.spatial(960, (((i_0_j_0_fused // 18 + 0) * 8 + i_2_j_2_fused // 32) * 8 + i_3) * 2 + i_4)
                                vj = T.axis.spatial(2304, ((i_0_j_0_fused % 18 * 2 + i_1_j_1_fused % 2) * 32 + i_2_j_2_fused % 32 + j_3) * 2 + j_4)
                                vk = T.axis.reduce(770, (k_0 + k_1) * 4 + k_2)
                                T.where((((i_0_j_0_fused // 18 + 0) * 8 + i_2_j_2_fused // 32) * 8 + i_3) * 2 + i_4 < 960 and (k_0 + k_1) * 4 + k_2 < 770)
                                T.reads(C_local[vi, vj], A_shared[vi, vk], B_shared[vk, vj])
                                T.writes(C_local[vi, vj])
                                T.block_attr({"meta_schedule.thread_extent_high_inclusive":1024, "meta_schedule.thread_extent_low_inclusive":32, "meta_schedule.tiling_structure":"SSSRRSRS"})
                                C_local[vi, vj] = C_local[vi, vj] + A_shared[vi, vk] * B_shared[vk, vj]
                    for ax0, ax1 in T.grid(16, 2):
                        with T.block("C_local"):
                            v0 = T.axis.spatial(960, i_0_j_0_fused // 18 * 128 + i_2_j_2_fused // 32 * 16 + ax0)
                            v1 = T.axis.spatial(2304, i_0_j_0_fused % 18 * 128 + i_1_j_1_fused * 64 + i_2_j_2_fused % 32 * 2 + ax1)
                            T.where(i_0_j_0_fused // 18 * 128 + i_2_j_2_fused // 32 * 16 + ax0 < 960)
                            T.reads(C_local[v0, v1])
                            T.writes(C[v0, v1])
                            C[v0, v1] = C_local[v0, v1]


if __name__ == '__main__':
    test_dense_cuda_sample_sched_infer()
