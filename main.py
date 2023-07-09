from Ecosystem import Ecosystem
import tensorflow as tf
import numpy as np


def run_loop():
    eco = Ecosystem([
        [16,16,64],
        [16,16,8],
        [16,16,4],
        [16,16,4],
        [64, 64],
        [32, 32],
        [32, 8],
        [8, 32],
        [8, 8],
        [4, 4],
        [2046],
        [1024],
        [1024],
        [128],
        [64],
        [4],
        [4],
        [4096]
    ]
    ,initial_cell_groups=200
    )
    eco.simulate(10000)


if "__main__" == __name__:
    test_array = tf.TensorArray(dtype=tf.float32, size=3)
    test_array = test_array.write(0, .5)
    test_array = test_array.write(1, .7)
    test_stack_1 = test_array.stack()
    test_array = test_array.write(2, 2.2)
    test_stack_2 = test_array.stack()
    run_loop()
