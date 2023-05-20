import tensorflow as tf
import numpy as np
import math
from enum import IntEnum
from typing import Tuple
from Field import Field

class ChromosomeSet(IntEnum):
    MAIN = 0
    RESET = 1
    FIELD_SHIFT = 2
    PROJECTION = 3
    EINSUM = 4
    CONV = 5
    MOVE = 6
    MATE = 7
    ADD_EPIGENE = 8
    REMOVE_EPIGENE = 9
    TRANSFER_ENERGY = 10
    RECEIVE_ENERGY = 11
    RECEPTORS = 12
    CONCAT = 13
    SELECT = 14
    DIVIDE = 15
    ADD = 16

class ActionSet(IntEnum):
    RESET = 0
    FIELD_SHIFT = 1
    PROJECTION = 2
    SOFTMAX = 3
    EINSUM = 4
    CONV = 5
    MOVE = 6
    MATE = 7
    ADD_EPIGENE = 8
    REMOVE_EPIGENE = 9
    LOCK = 10
    UNLOCK = 11
    TRANSFER_ENERGY = 12
    WAIT = 13
    CONCAT = 14
    DIVIDE = 15
    ADD = 16


class Cell:
    def __init__(self, fields : Tuple[Field], mother_genome, father_genome, mutation_rate, epigenetics, epigene_manifest, w, x, y, z):
        self.fields = fields
        self._field_index = -1
        self.graph_ops = []
        self.chromosomes = []
        for mother_chromosome, father_chromosome in zip(mother_genome, father_genome):
            chromosome = []
            for mother_gene, father_gene in zip(mother_chromosome, father_chromosome):
                mutation = tf.random.normal(tf.shape(mother_gene), mean=0.0, stddev=mutation_rate)
                chromosome.append(
                    (mother_gene + father_gene) / 2 + mutation
                )
            self.chromosomes.append(chromosome)
        self.epigenetics = epigenetics
        self._energy = 100.0
        self.epigene_manifest = epigene_manifest
        self._action_mask = tf.constant([1] + [0] * 15, dtype=tf.float32)
        self.w = w
        self.x = x
        self.y = y
        self.z = z
        self._last_reward = 0
        self._gradient_acceptor = tf.Variable(1.0)

    def provide_chromosome(self, chromosome_index):
        return self.chromosomes[chromosome_index]

    def provide_epigenes(self, gene_layer):
        return self.epigenetics[gene_layer]

    @property
    def gradient_acceptor(self):
        return self._gradient_acceptor

    @property
    def energy(self):
        return self._energy

    @property
    def last_reward(self):
        return self._last_reward

    @property
    def field_index(self):
        return self._field_index

    @property
    def action_mask(self):
        return tf.multiply(
            self._action_mask,
            self.current_output_field.action_mask
        )

    def reward(self, update):
        self._last_reward = update
        self._energy += update

    @property
    def is_dead(self):
        return self._energy <= 0

    @property
    def current_output_field(self):
        return self.fields[self._field_index]

    def accept_environment_epigene_manifext(self, manifest):
        self.epigene_manifest = manifest

    def add_field_shift(self, dimension_to):
        self.graph_ops.append(self.current_output_field.dimensional_shift[dimension_to])
        self._field_index = dimension_to
        self._energy -= 1

    def add_projection(self, dimension_to, projection_key):
        self.graph_ops.append(self.current_output_field.add_projection[dimension_to](projection_key))
        self._field_index = dimension_to
        self._energy -= 3

    def add_softmax(self):
        self.graph_ops.append(self.current_output_field.softmax)
        self._energy -= 5

    def add_einsum(self, dimension_to, dimension_with):
        self.graph_ops.append(self.current_output_field.build_einsum_op(dimension_with, dimension_to))
        self._field_index = dimension_to
        self._energy -= 20

    def add_conv(self, dimension_to, conv_gen_state):
        self.graph_ops.append(self.current_output_field.make_conv[dimension_to](conv_gen_state))
        self._field_index = dimension_to
        self._energy -= 10

    def move(self, delta_x, delta_y):
        self.x += delta_x
        self.y += delta_y
        self._energy -= math.ceil(math.sqrt(delta_x ** 2 + delta_y ** 2))

    def mate(self):
        if 100 < self._energy:
            self._energy -= 100
            return True
        else:
            return False

    def update_epigene(self, gene_layer, index, value):
        update_var = np.array(self.epigenetics[gene_layer].value)
        update_var[index] = value
        self.epigenetics[gene_layer].assign(update_var)

    def add_epigene(self, gene_layer, index):
        self.update_epigene(gene_layer, index, 0.0)
        self._energy -= 1

    def subtract_epigene(self, gene_layer, index_heat_map):
        filtered_index_heat_map = tf.multiply(index_heat_map, self.epigenetics[gene_layer])
        index = tf.argmax(filtered_index_heat_map)
        self.update_epigene(gene_layer, index, 1.0)
        self._energy -= 1

    def lock(self):
        self._action_mask = tf.constant([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0], dtype=tf.float32)

    def unlock(self):
        self._action_mask = tf.constant([1] * 15 + [0], dtype=tf.float32)

    def accept_energy(self, sender_key):
        hidden = tf.nn.relu(
            tf.matmul(sender_key, self.chromosomes[ChromosomeSet.RECEIVE_ENERGY][0]) +
            self.chromosomes[ChromosomeSet.RECEIVE_ENERGY][1]
        )
        energy_function_params = tf.matmul(hidden, self.chromosomes[ChromosomeSet.RECEIVE_ENERGY][2]) +\
            self.chromosomes[ChromosomeSet.RECEIVE_ENERGY][3]
        energy_scale = tf.math.cos(energy_function_params[0] / energy_function_params[1])
        energy_transfer = energy_function_params[0] * energy_scale
        if energy_transfer > self._energy:
            energy_transfer = self._energy
        self._energy -= energy_transfer
        return energy_transfer

    def reset(self, starting_field):
        self._field_index = starting_field
        self.graph_ops = []
        pass

    def add_concat(self, dimension_with):
        concat_op, dimension_to = self.current_output_field.make_concat[dimension_with]
        self.graph_ops.append(concat_op)
        self._field_index = dimension_to
        self._energy -= 5

    def add_divide(self, dimension_with):
        divide_op, dimension_to = self.current_output_field.make_divide[dimension_with]
        self.graph_ops.append(divide_op)
        self._field_index = dimension_to
        self._energy -= 10

    def add_add(self, dimension_with):
        add_op, dimension_to = self.current_output_field.make_add[dimension_with]
        self.graph_ops.append(add_op)
        self._field_index = dimension_to
        self._energy -= 10

    def generate_ops(self):
        @tf.function
        def ops(cell_input_field_values):
            cell_op_state = [tf.identity(field_value) for field_value in cell_input_field_values]
            for op in self.graph_ops:
                op(cell_op_state)
            return cell_op_state[self._field_index]