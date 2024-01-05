import os

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
    # , initial_cell_groups=200
    )
    simulation_steps = 1000
    # simulation_steps = 5
    eco.simulate(
        simulation_steps,
        r"/media/abyssal-dragon/New Volume/ArtAssistData/Records/OriginalIntermediateLayers.tfrecord",
        "/media/abyssal-dragon/New Volume/ArtAssistData/Records/generate_graph.txt",
        energy_reward=2000)
    # eco.test_scaling()


if "__main__" == __name__:
    # Ensures the records drive is mounted.
    _ = os.listdir("/media/abyssal-dragon/New Volume/ArtAssistData/Records/")

    run_loop()
