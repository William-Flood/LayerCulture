import tensorflow as tf
import numpy as np
import math
from enum import IntEnum

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


class Cell:
    def __init__(self, fields, mother_genome, father_genome, mutation_rate, epigenetics, epigene_manifest, x, y):
        self.fields = fields
        self._field_index = -1
        self.graph_ops = []
        self.chromosomes = []
        for mother_chromosome, father_chromosome in zip(mother_genome, father_genome):
            mutation = tf.random.normal(tf.shape(mother_chromosome), mean=0.0, stddev=mutation_rate)
            self.chromosomes.append(
                (mother_chromosome + father_chromosome) / 2 + mutation
            )
        self.epigenetics = epigenetics
        self.energy = 100.0
        self.epigene_manifest = epigene_manifest
        self._action_mask = tf.constant([1] + [0] * 15, dtype=tf.float32)
        self.x = x
        self.y = y

    def provide_chromosome(self, chromosome_index):
        return self.chromosomes[chromosome_index]

    def provide_epigenes(self, gene_layer):
        return self.epigenetics[gene_layer]

    @property
    def field_index(self):
        return self._field_index

    @property
    def action_mask(self):
        return tf.multiply(
            self._action_mask,
            self.fields[self._field_index].action_mask
        )

    @property
    def is_dead(self):
        return self.energy <= 0

    def accept_environment_epigene_manifext(self, manifest):
        self.epigene_manifest = manifest

    def add_field_shift(self, dimension_to):
        self.graph_ops.append(self.fields[self._field_index].dimensional_shift[dimension_to])
        self._field_index = dimension_to
        self.energy -= 1

    def add_projection(self, dimension_to, projection_key):
        self.graph_ops.append(self.fields[self._field_index].add_projection[dimension_to](projection_key))
        self._field_index = dimension_to
        self.energy -= 3

    def add_softmax(self):
        self.graph_ops.append(self.fields[self._field_index].softmax)
        self.energy -= 5

    def add_einsum(self, dimension_to, einsum_gen_state):
        self.graph_ops.append(self.fields[self._field_index].einsum(dimension_to, einsum_gen_state))
        self._field_index = dimension_to
        self.energy -= 20

    def add_conv(self, dimension_to, conv_gen_state):
        self.graph_ops.append(self.fields[self._field_index].make_conv[dimension_to](conv_gen_state))
        self._field_index = dimension_to
        self.energy -= 10

    def move(self, delta_x, delta_y):
        self.x += delta_x
        self.y += delta_y
        self.energy -= math.ceil(math.sqrt(delta_x ** 2 + delta_y ** 2))

    def mate(self):
        if 100 < self.energy:
            self.energy -= 100
            return True
        else:
            return False

    def update_epigene(self, gene_layer, index, value):
        update_var = np.array(self.epigenetics[gene_layer].value)
        update_var[index] = value
        self.epigenetics[gene_layer].assign(update_var)

    def add_epigene(self, gene_layer, index):
        self.update_epigene(gene_layer, index, 0.0)
        self.energy -= 1

    def subtract_epigene(self, gene_layer, index_heat_map):
        filtered_index_heat_map = tf.multiply(index_heat_map, self.epigenetics[gene_layer])
        index = tf.argmax(filtered_index_heat_map)
        self.update_epigene(gene_layer, index, 1.0)
        self.energy -= 1

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
        self.energy -= energy_transfer
        return energy_transfer

    def reset(self, starting_field):
        self._field_index = starting_field
        self.graph_ops = 0
        pass
