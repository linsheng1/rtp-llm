"""Stdlib-only fixed decode-slot oracle and integer-kernel CPU contracts.

The lane interpreter executes the actual Triton function's Python AST, not a
second implementation of its index algorithm. It checks integer layout only;
Triton lowering, device stores, and FlashMLA numerics still require GPU tests.
"""

import ast
import copy
import itertools
import operator
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "_v41_prefill_index_plan.py"


def int32(value):
    return (value + 2**31) % 2**32 - 2**31


oracle_source = Path(__file__).with_name("test_v41_prefill_index_plan.py")
oracle_tree = ast.parse(oracle_source.read_text())
oracle_namespace = {"int32": int32}
oracle_function = next(n for n in oracle_tree.body if isinstance(n, ast.FunctionDef)
                       and n.name == "decode_slot_reference_row")
exec(compile(ast.Module(body=[oracle_function], type_ignores=[]), str(oracle_source), "exec"), oracle_namespace)
decode_slot_reference_row = oracle_namespace["decode_slot_reference_row"]


class Lanes:
    def __init__(self, values):
        self.values = values if isinstance(values, list) else [values]

    def at(self, index):
        return self.values[0 if len(self.values) == 1 else index]

    def to(self, dtype):
        return Lanes([int32(v) if dtype == "i32" else int(v) for v in self.values])

    def binary(self, other, op):
        other = lanes(other)
        return Lanes([op(self.at(i), other.at(i))
                      for i in range(max(len(self.values), len(other.values)))])


def lanes(value):
    return value if isinstance(value, Lanes) else Lanes(value)


for name, function in (("__add__", operator.add), ("__sub__", operator.sub),
                       ("__mul__", operator.mul), ("__lt__", operator.lt),
                       ("__le__", operator.le), ("__ge__", operator.ge),
                       ("__and__", operator.and_)):
    setattr(Lanes, name, lambda self, other, op=function: self.binary(other, op))


class Pointer:
    def __init__(self, memory, offset=0, dtype="i64"):
        self.memory, self.offset = memory, lanes(offset)
        self.dtype = SimpleNamespace(element_ty=dtype)

    def __add__(self, offset):
        return Pointer(self.memory, self.offset + offset, self.dtype.element_ty)


class IntegerTL:
    int32, int64, constexpr = "i32", "i64", int

    def program_id(self, axis):
        return Lanes(self.row)

    def arange(self, first, last):
        return Lanes(list(range(first, last)))

    def load(self, pointer, mask=True, other=0):
        mask, other = lanes(mask), lanes(other)
        return Lanes([pointer.memory[int(index)] if mask.at(i) else other.at(i)
                      for i, index in enumerate(pointer.offset.values)])

    def store(self, pointer, value, mask=True):
        value, mask = lanes(value), lanes(mask)
        for i, index in enumerate(pointer.offset.values):
            if mask.at(i):
                pointer.memory[int(index)] = value.at(i)

    def maximum(self, left, right):
        return lanes(left).binary(right, max)

    def where(self, condition, yes, no):
        condition, yes, no = lanes(condition), lanes(yes), lanes(no)
        return Lanes([yes.at(i) if condition.at(i) else no.at(i)
                      for i in range(max(len(v.values) for v in (condition, yes, no)))])

    def sort(self, value, descending):
        return Lanes(sorted(value.values, reverse=descending))

    def cumsum(self, value, axis):
        return Lanes(list(itertools.accumulate(value.values)))

    def sum(self, value, axis):
        return Lanes(sum(value.values))


def execute_integer_kernel(selected, positions, offsets, counts, starts, window,
                           *, tensor_meta=False, dtype="i32"):
    tree = ast.parse(SOURCE.read_text())
    function = copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                                  and n.name == "_prefill_index_plan_kernel"))
    function.decorator_list = []
    tl = IntegerTL()
    namespace = {"tl": tl}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(SOURCE), "exec"), namespace)
    rows, k = len(selected), len(selected[0])
    padded = (window + k + 63) // 64 * 64
    selected_memory = [0, 0] + list(itertools.chain.from_iterable(row + [0] * 3 for row in selected))
    position_memory = [0] + list(itertools.chain.from_iterable([p, 0] for p in positions))
    output, lengths = [None] * (rows * padded), [None] * rows

    def metadata(values):
        if not tensor_meta:
            return Lanes(values)
        return Pointer([0] + list(itertools.chain.from_iterable([v, 0, 0] for v in values)), 1)

    for row in range(rows):
        tl.row = row
        namespace[function.name](Pointer(selected_memory, 2, dtype), Pointer(position_memory, 1),
            Pointer(output), Pointer(lengths), metadata(offsets), metadata(counts), metadata(starts),
            k + 3, 2, k, window, padded, 1 << (padded - 1).bit_length(), 1 << (k - 1).bit_length(),
            tensor_meta, 3, 3, 3)
    return [output[r * padded:(r + 1) * padded] for r in range(rows)], lengths


class FixedSlotCPU(unittest.TestCase):
    def check(self, selected, positions, offsets, counts, starts, window=128, tensor_meta=False, dtype="i32"):
        actual, lengths = execute_integer_kernel(selected, positions, offsets, counts, starts, window,
                                                 tensor_meta=tensor_meta, dtype=dtype)
        for row, values in enumerate(selected):
            meta = [v[row] if tensor_meta else v for v in (offsets, counts, starts)]
            expected, extent = decode_slot_reference_row(values, positions[row], *meta, window)
            self.assertEqual(actual[row], expected)
            self.assertEqual(lengths[row], extent)
        return actual, lengths

    def test_decode_boundaries_workspace_holes_and_global_valid_counts(self):
        for position, valid in itertools.product((0, 63, 64, 126, 127, 128), (0, 1, 511, 512)):
            selected = list(reversed(list(range(valid)))) + [-7] * (512 - valid)
            for start in (0, max(position - 127, 0) + 1, position + 1):
                with self.subTest(position=position, valid=valid, start=start):
                    self.check([selected], [position], 11, 4096, start)

    def test_no_cross_pool_compaction_or_global_truncation(self):
        rows, lengths = self.check([[9, -3, 2, 9, -1]], [4], 10, 20, 3, window=4)
        self.assertEqual(rows[0][:9], [-1, -1, 30, 31, 12, 19, 19, -1, -1])
        self.assertEqual(lengths, [9])
        # SWA slot zero remains absent; GLOBAL starts at W even with no SWA.
        rows, _ = self.check([[7] + [-1] * 511], [0], 0, 4096, 1)
        self.assertEqual(rows[0][:128], [-1] * 128)
        self.assertEqual(rows[0][128], 7)

    def test_ragged_cp_offsets_duplicates_negative_and_strided_tensor_metadata(self):
        selected = [[7, -9, 2, 7, -1], [-3] * 5, [4, 4, 0, -1, 9]]
        self.check(selected, [64, 0, 128], [7, 200, 8192], [4096, 0, 512], [1, 1, 63],
                   window=127, tensor_meta=True)

    def test_integer_width_casts_stay_in_global_region_and_padding_is_not_extent(self):
        self.check([[2**31 - 1, -9, 3], [2**40, 2**31, -1]],
                   [2**31, -(2**40)], 1, 7, 0, window=7, dtype="i64")
        rows, lengths = self.check([[3]], [0], 0, 8, 0, window=1)
        self.assertEqual(rows[0][:2], [8, 3])
        self.assertEqual(rows[0][2:], [-1] * 62)
        self.assertEqual(lengths, [2])


if __name__ == "__main__":
    unittest.main()
