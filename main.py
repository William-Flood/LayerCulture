from Ecosystem import Ecosystem
import tensorflow as tf


def run_loop():
    eco = Ecosystem([
        [16,16,64],
        [4096],
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
        [4]
    ])


if "__main__" == __name__:
    run_loop()
