# Copyright 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
import numpy as np
from npu.build.kernel import Kernel
from npu.build.appbuilder import AppBuilder
from npu.runtime import AppRunner
from .test_applications import check_npu
from ml_dtypes import bfloat16


kernel_src = '''
#include <aie_api/aie.hpp>

extern "C" {

void vector_add_factor(uint8_t *in_buffer, uint8_t *out_buffer, uint8_t scale) {
    auto vec_in = ::aie::load_v<32>(in_buffer);
    in_buffer += 32;
    auto out = ::aie::add(vec_in, (uint8_t) scale);
    ::aie::store_v(out_buffer, out);
    out_buffer += 32;
}

} // extern "C"
'''


def vector_add_factor_behavior(invobj):
    invobj.out_buffer.array = invobj.in_buffer.array


def _appbuild_and_test(datatype, interpret_bits = True):
    check_npu()
    shape = (32, 1)
    bpi = np.dtype(datatype).itemsize
    buffin = np.zeros(shape=shape, dtype=datatype)
    buffout = np.zeros(buffin.shape, dtype=buffin.dtype)

    datatype_txt = str(np.dtype(datatype))
    kernel_src0 = kernel_src.replace('uint8_t', f'{datatype_txt}' +
                                      ('' if datatype == bfloat16 else '_t'))

    vecadd_factor = Kernel(kernel_src0, vector_add_factor_behavior)

    class SingleBufferOneRtp(AppBuilder):
        def __init__(self, kernel):
            super().__init__()
            self.kernel = kernel

        def callgraph(self, x_in, x_out):
            x_out[:] = self.kernel(x_in, 1)

    trace_app = SingleBufferOneRtp(vecadd_factor)
    trace_app.build(buffin, buffout)

    app = AppRunner(f"{trace_app.name}.xclbin")

    # generate a numpy array of random data of shape buffin.shape
    if datatype in [bfloat16, np.float32]:
        test_data = bfloat16(np.random.randn(*buffin.shape))
        value = datatype(1.136)
        if interpret_bits:
            factor = int.from_bytes(value.tobytes(), "big")
        else:
            factor = value
    else:
        test_data = np.random.randint(0, (2**(bpi*8))-1, buffin.shape,
                                      dtype=buffin.dtype)
        factor = 3

    res = np.zeros(buffin.shape, dtype=buffin.dtype)
    bo_in = app.allocate(shape=buffin.shape, dtype=buffin.dtype)
    bo_out = app.allocate(shape=buffin.shape, dtype=buffin.dtype)

    bo_in[:] = test_data
    bo_in.sync_to_npu()
    app.vector_add_factor_0.scale = factor
    app.call(bo_in, bo_out)
    bo_out.sync_from_npu()
    res[:] = bo_out[:]

    del app
    assert np.array_equal(test_data + datatype(factor), res)


@pytest.mark.parametrize('datatype', [np.uint8, np.uint16, np.uint32,
                                      np.float32, bfloat16])
def test_appbuild_rtp_datatype(datatype):
    _appbuild_and_test(datatype)


@pytest.mark.xfail(reason="Float datatypes are not properly interpreted")
@pytest.mark.parametrize('datatype', [np.float32, bfloat16])
def test_appbuild_rtp_datatype_fail(datatype):
    _appbuild_and_test(datatype, interpret_bits=False)
