## https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md
from numbers import Number

import numpy as np
from fvcore.nn import FlopCountAnalysis
from typing import Any, Callable, List, Optional, Union, Counter, Dict


def mul_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for element wise multiplication.
    """
    # inputs should be a list of at least length 1.
    # inputs contains the shape of the matrix.
    input_shapes = [v.type().sizes() for v in inputs]
    output_shapes = [v.type().sizes() for v in outputs]

    # print('Mul in {} ----- out {}'.format(input_shapes, output_shapes))

    assert len(input_shapes) >= 1, input_shapes
    flop = 0.5 * np.prod(input_shapes[0]) # larger of the input shapes
    return flop

def add_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for add. Handles both sum and element-wise addition.
    """
    # inputs should be a list of at least length 1.
    # inputs contains the shape of the matrix.
    input_shapes = []
    for v in inputs:
        if str(v.type()) == 'Tensor':
            input_shapes.append(v.type().sizes())

    output_shapes = []
    for v in outputs:
        if str(v.type()) == 'Tensor':
            input_shapes.append(v.type().sizes())

    assert len(input_shapes) >= 1, input_shapes
    # print('Add in {} ----- out {}'.format(input_shapes, output_shapes))

    flop = 0.5 * np.prod(input_shapes[0])  ## will be 2x at the end, so we divide by 2 here to balance it out
    return flop



def compute_flops(model, inputs):
    Handle = Callable[[List[Any], List[Any]], Union[Counter[str], Number]]
    ops: Dict[str, Handle] = {
        "aten::mul": mul_flop_jit,
        "aten::add": add_flop_jit,
        "aten::sum": add_flop_jit,
        "aten::avg_pool2d": add_flop_jit,
        "aten::max_pool2d": add_flop_jit,
        "aten::batch_norm": None,  # Can be fused at inference time so ignore (matches Slimmable)
        "aten::add_":  None,
    }

    ops = FlopCountAnalysis(model, inputs).set_op_handle(**ops)
    flops = ops.total() * 2
    return flops


def compute_flops_with_members(model, inputs, member_id: Union[List[int], None]):
    if member_id is None:
        flops = compute_flops(model, (inputs, member_id))
    else:
        flops = 0
        for member in member_id:
            flops += compute_flops(model, (inputs, member))

    return flops / 1e9
