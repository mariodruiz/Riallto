# Copyright 2023 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


import cv2
import numpy as np
from pathlib import Path
from npu.build.kernel import Kernel
import npu.runtime as ipr
from npu.runtime import AppRunner
from .test_applications import check_npu, manage_testing, SimplePlusN, AppITTilingPlusN
from .test_applications import Kern2KernDirectMemITTIlingRGBAInverse, SimpleKernelITTiling, SimpleGray2RGBA


def test_plus_n_1_dim(manage_testing):
    """End-to-end build and run test for a single PlusN kernel. """
    check_npu()
    array = np.zeros(shape=(256), dtype=np.uint8)
    n = 5

    trace_app = SimplePlusN()
    trace_app.build(array, n)

    app = AppRunner(f"{trace_app.name}.xclbin")

    test_data = np.random.randint(0, 255, 256, dtype=np.uint8)
    bo_in = app.allocate(shape=(256), dtype=np.uint8)
    bo_out = app.allocate(shape=(256), dtype=np.uint8)

    bo_in[:] = test_data
    bo_in.sync_to_npu()

    app._refresh_sequence()
    app.call(bo_in, bo_out)

    bo_out.sync_from_npu()
    test_out = np.array(bo_out)

    del app
    assert np.array_equal(test_data+n, test_out)


def test_plus_n_rtp(manage_testing):
    """End-to-end build and run test for a single PlusN kernel, with a RTP configured. """
    check_npu()

    ain = np.zeros(shape=(10, 256, 32),dtype=np.uint8)
    aout = np.zeros(shape=(10, 256, 32),dtype=np.uint8)

    n = 5
    trace_app = AppITTilingPlusN()
    trace_app.build(ain, aout, n)

    app = AppRunner(f"{trace_app.name}.xclbin")

    test_data = np.random.randint(0, 255,(10, 256, 32), dtype=np.uint8)
    bo_in = app.allocate(shape=(10, 256, 32), dtype=np.uint8)
    bo_out = app.allocate(shape=(10, 256, 32), dtype=np.uint8)

    bo_in[:] = test_data
    bo_in.sync_to_npu()

    app._refresh_sequence()
    app.call(bo_in, bo_out)

    bo_out.sync_from_npu()
    test_out = np.array(bo_out).reshape(10, 256, 32)

    assert np.array_equal(test_data+n,test_out)

    del app


def test_gray2rgba_kernelio_resizing(manage_testing):
    """End-to-end build and run test where input and output buffers have a different shape after being passed through Gray2RGBA kernel. """
    check_npu()

    test_data = np.random.randint(0, 255, 256,dtype=np.uint8)
    test_data = test_data.reshape(64, 4)

    appbuilder = SimpleGray2RGBA()
    appbuilder.build(test_data)
    app = AppRunner(f"{appbuilder.name}.xclbin")

    bo_in = app.allocate(shape=(64, 4), dtype=np.uint8)
    bo_out = app.allocate(shape=(256, 4), dtype=np.uint8)

    bo_in[:] = test_data
    bo_in.sync_to_npu()

    app.call(bo_in, bo_out)

    bo_out.sync_from_npu()
    test_out = np.array(bo_out).reshape(64, 4, 4)

    opencv_test_out = cv2.cvtColor(test_data, cv2.COLOR_GRAY2RGBA)

    del app

    assert np.array_equal(test_out, opencv_test_out)


ident_src = '''
#define IMAGE_WIDTH 256
#include <aie_api/aie.hpp>
extern "C" {
void ident(uint8_t *in_buffer, uint8_t* out_buffer)
{
    ::aie::vector<uint8_t, 32> buffer;
    uint16_t loop_count = (IMAGE_WIDTH*4) >> 5;
    for(int j=0; j<loop_count; j++) {
        buffer = ::aie::load_v<32>(in_buffer);
        in_buffer += 32;
        ::aie::store_v((uint8_t*)out_buffer, buffer);
        out_buffer += 32;
    }
}
} // extern "C"
'''


def ident_behavior(invobj):
    invobj.out_buffer.array = invobj.in_buffer.array


def test_ident_shim_rgba_pyxrt_notebook(manage_testing):
    """End-to-end build and run test using the pyxrt bindings instead of AppRunner class."""
    check_npu()

    ident = Kernel(ident_src, ident_behavior)
    ident.build()

    imgbuffer_in = np.zeros(shape=(256, 256, 4),dtype=np.uint8)
    imgbuffer_out = np.ones(shape=(256, 256, 4),dtype=np.uint8)

    trace_app = SimpleKernelITTiling(ident)
    trace_app.callgraph(imgbuffer_in, imgbuffer_out)
    assert np.array_equal(imgbuffer_in, imgbuffer_out)
    assert imgbuffer_in.nbytes == imgbuffer_out.nbytes == 256*256*4

    trace_app.build(imgbuffer_in, imgbuffer_out)

    device = ipr.device(0)
    xclbin = ipr.xclbin(f"{trace_app.name}.xclbin")

    kernels = xclbin.get_kernels()
    for kernel in kernels:
        if kernel.get_name() == trace_app.name:
            xkernel = kernel

    kernel_name = xkernel.get_name()

    uuid = device.register_xclbin(xclbin)
    context = ipr.hw_context(device, uuid)

    kern = ipr.kernel(context, kernel_name)

    seq = ipr.Sequence(f"{trace_app.name}.seq")

    instr = ipr.bo(device, seq.buffer.nbytes, ipr.bo.flags.cacheable, kern.group_id(0))
    bo_in = ipr.bo(device, 256*256*4, ipr.bo.flags.host_only, kern.group_id(2))
    bo_out = ipr.bo(device, 256*256*4, ipr.bo.flags.host_only, kern.group_id(3))

    img_bgr = cv2.imread(Path('tests/images/test_image.png').as_posix())
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb_cropped = img_rgb[0:256, 0:256]
    alpha_channel =np.full((img_rgb_cropped.shape[0], img_rgb_cropped.shape[1], 1), 255, dtype=np.uint8)
    img_rgba = np.concatenate((img_rgb_cropped, alpha_channel), axis=-1)

    bo_in.write(img_rgba, 0)
    instr.write(seq.buffer, 0)

    instr.sync(ipr.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_in.sync(ipr.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    run = kern(instr, len(seq.buffer), bo_in, bo_out)
    run.wait()
    bo_out.sync(ipr.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, bo_out.size(), 0)

    test_out = np.array(bo_out.map(), dtype=np.uint8).reshape(256, 256, 4)

    del bo_in
    del instr
    del bo_out
    del context
    del kern
    del device

    assert np.array_equal(test_out, img_rgba)


def test_ident_shim_rgba_one_notebook(manage_testing):
    """End-to-end build and run test that uses a simple identity kernel and the interface tile directly."""
    check_npu()

    ident = Kernel(ident_src, ident_behavior)
    ident.build()

    imgbuffer_in = np.zeros(shape=(256, 256, 4),dtype=np.uint8)
    imgbuffer_out = np.ones(shape=(256, 256, 4),dtype=np.uint8)

    trace_app = SimpleKernelITTiling(ident)
    trace_app.callgraph(imgbuffer_in, imgbuffer_out)

    assert np.array_equal(imgbuffer_in, imgbuffer_out)

    trace_app.build(imgbuffer_in, imgbuffer_out)

    app = AppRunner(f"{trace_app.name}.xclbin")

    bo_in = app.allocate(shape=(256, 256, 4), dtype=np.uint8)
    bo_out = app.allocate(shape=(256, 256, 4), dtype=np.uint8)

    img_bgr = cv2.imread(Path('tests/images/test_image.png').as_posix())
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb_cropped = img_rgb[0:256, 0:256]
    alpha_channel =np.full((img_rgb_cropped.shape[0], img_rgb_cropped.shape[1], 1), 255, dtype=np.uint8)
    img_rgba = np.concatenate((img_rgb_cropped, alpha_channel), axis=-1)

    assert img_rgba.shape == bo_in.shape

    bo_in[:] = img_rgba
    bo_in.sync_to_npu()
    app.call(bo_in, bo_out)
    bo_out.sync_from_npu()

    test_out = np.array(bo_out)

    del app

    assert np.array_equal(test_out, img_rgba)


def test_double_inverse_chain(manage_testing):
    """End-to-end build and run test that chains two inverse (uint8) kernels together and ensures parity on input and output."""
    check_npu()

    ain = np.zeros(shape=(10, 256),dtype=np.uint8)
    aout = np.zeros(shape=(10, 256),dtype=np.uint8)

    trace_app = Kern2KernDirectMemITTIlingRGBAInverse()
    trace_app.build(ain, aout)

    app = AppRunner(f"{trace_app.name}.xclbin")

    test_data = np.random.randint(0, 255, (10, 256))
    bo_in = app.allocate(shape=(10, 256), dtype=np.uint8)
    bo_out = app.allocate(shape=(10, 256), dtype=np.uint8)

    bo_in[:] = test_data
    bo_in.sync_to_npu()

    app.call(bo_in, bo_out)

    bo_out.sync_from_npu()
    test_out = np.array(bo_out)

    del app

    assert np.array_equal(test_data, test_out)
