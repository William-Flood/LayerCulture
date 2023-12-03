import numpy as np

from Graph import Graph
import tensorflow as tf
import DatasetProvider
from Field import Field
from multiprocessing import Queue
from numpy.typing import NDArray


def train(cell_export_data: NDArray, fields_shapes, graph_training_cycles,
                distance_scalings: NDArray, energy_reward,
                training_set_file_name, output_queue: Queue, last_golden):
    test_var = tf.constant(6)
    fields = tuple(Field(fields_shape, field_index) for field_index, fields_shape in enumerate(fields_shapes))
    for field in fields:
        field.build_graphs(fields)
    training_set_iter = DatasetProvider.provide_dataset(training_set_file_name, batch_size=64)
    trainer = tf.keras.optimizers.RMSprop()
    training_graph = Graph(cell_export_data, fields, 5, 1000, distance_scalings, last_golden)
    if 0 < len(training_graph.hot_sources):
        output_queue.put("Training")
        times_array, losses_array, result_graph = training_graph.train(training_set_iter, trainer, graph_training_cycles, 5, .2, .25)
        output_queue.put(times_array)
        output_queue.put(losses_array)
        output_queue.put(result_graph)
        # print(f"Losses went from {losses_array[0]} to {losses_array[-1]}")
    else:
        output_queue.put("Skipping")
        output_queue.put([])
        output_queue.put([])
        output_queue.put([False] * cell_export_data.shape[0])
    output_queue.put(training_graph.calculate_rewards(energy_reward, distance_scalings))
    # try:
    # except BaseException as e:
    #     output_queue.put("Error")
    #     output_queue.put([])
    #     output_queue.put([])

