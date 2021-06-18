import torch
from torch.testing._internal.common_device_type import (
    ops, OpDTypes, instantiate_device_type_tests, dtypes, deviceCountAtLeast)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import TestCase, run_tests, parametrize, instantiate_parametrized_tests
from torch.testing import floating_types


class TestBlah(TestCase):
    @parametrize("x", range(10))
    def test_something(self, x):
        print('Passed in:', x)

    @parametrize("x,y", [(1, 2), (3, 4), (5, 6)])
    def test_two_things(self, x, y):
        print('Passed in:', x, y)

    @parametrize("dtype", floating_types())
    def test_something_with_dtypes(self, dtype):
        print('Passed in:', dtype)


class TestDeviceBlah(TestCase):
    @parametrize("x", range(10))
    def test_something_with_device(self, device, x):
        print('Passed in:', device, x)

    @parametrize("x,y", [(1, 2), (3, 4), (5, 6)])
    def test_two_things_with_device(self, device, x, y):
        print('Passed in:', device, x, y)

    @parametrize("dtype", floating_types())
    def test_something_with_device_and_dtype(self, device, dtype):
        print('Passed in:', device, dtype)

    @deviceCountAtLeast(1)
    def test_something_with_devices(self, devices):
        print('Passed in:', devices)


instantiate_parametrized_tests(TestBlah)
instantiate_device_type_tests(TestDeviceBlah, globals())


if __name__ == '__main__':
    run_tests()
