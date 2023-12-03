import tensorflow as tf
import numpy as np
import math
from typing import Tuple
from Field import Field
from CellConstants import ActionSet, ChromosomeSet, OP_GEN_STATE_SIZE
from cell_export_manager import cell_data_to_array


def breed_cells(mother, father, mutation_rate):
    chromosomes = []
    for mother_chromosome, father_chromosome in zip(mother.chromosomes, father.chromosomes):
        chromosome = []
        for mother_gene, father_gene in zip(mother_chromosome, father_chromosome):
            mutation = tf.random.normal(tf.shape(mother_gene), mean=0.0, stddev=mutation_rate)
            chromosome.append(
                (mother_gene + father_gene) / 2 + mutation
            )
        chromosomes.append(chromosome)
    return chromosomes


class Cell:
    def __init__(self, fields: Tuple[Field], chromosomes, epigenetics, w, x, y, z):
        pass
        self.fields = fields
        self.chromosomes = chromosomes
        self._epigenetics = epigenetics
        self._energy = 100.0
        self._action_mask = []
        self.unlock()
        self._gradient_acceptor = tf.Variable(1.0)
        self.w = w
        self.x = x
        self.y = y
        self.z = z
        self._last_reward = 0
        self._transmit_reward = 0
        self._receive_reward = 0
        self._is_locked = False
        self.golden_lock = False
        self.graph_op_inputs = []
        self.param_count = 0
        self.left_index = -1
        self.right_index = -1
        self.action_type = -1
        self.return_index = -1
        self.generation_state = np.zeros([OP_GEN_STATE_SIZE], dtype=np.float32)

    def provide_chromosome(self, chromosome_index):
        return self.chromosomes[chromosome_index]

    @property
    def epigenetics(self):
        return self._epigenetics

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
    def action_mask(self):
        if self.golden_lock:
            golden_mask = np.zeros([len(ActionSet)], dtype=np.float32)
            golden_mask[ActionSet.MATE] = 1.0
            golden_mask[ActionSet.ADD_EPIGENE] = 1.0
            golden_mask[ActionSet.REMOVE_EPIGENE] = 1.0
            golden_mask[ActionSet.TRANSFER_ENERGY] = 1.0
            golden_mask[ActionSet.WAIT] = 1.0
            return golden_mask
        else:
            return self._action_mask

    def reward(self, update):
        self._last_reward = update
        self._energy += update

    def conclude_transmit(self, transmitted_energy):
        self._energy += transmitted_energy
        self._transmit_reward = transmitted_energy

    @property
    def last_transmit_reward(self):
        return self._transmit_reward

    @property
    def last_receive_reward(self):
        return self._receive_reward

    @property
    def is_dead(self):
        return self._energy <= 0 and not self.golden_lock


    @property
    def is_locked(self):
        return self._is_locked

    @property
    def has_graph(self):
        return self.action_type != -1

    @property
    def position(self):
        return np.array([self.w, self.x, self.y, self.z], dtype=np.float32)

    @property
    def op_data(self):
        left_field_component = np.zeros([len(self.fields)], dtype=np.float32)
        if self.left_index is not None:
            left_field_component[self.left_index] = 1.0
        right_field_component = np.zeros([len(self.fields)], dtype=np.float32)
        if self.right_index is not None:
            right_field_component[self.right_index] = 1.0
        return_field_component = np.zeros([len(self.fields)], dtype=np.float32)
        if self.return_index is not None:
            return_field_component[self.return_index] = 1.0
        action_component = np.zeros([len(ActionSet)], dtype=np.float32)
        action_component[self.action_type] = 1.0
        return np.concatenate([
            left_field_component,
            right_field_component,
            return_field_component,
            action_component,
            self.generation_state
        ])

    def accept_environment_epigene_manifext(self, manifest):
        self.epigene_manifest = manifest

    def add_field_shift(self, operating_dimension, dimension_to):
        if self.golden_lock:
            raise Exception("Invalid operation on golden cell")
        self.left_index = operating_dimension
        self.right_index = -1
        self.return_index = dimension_to
        self.action_type = ActionSet.FIELD_SHIFT
        self.graph_op_inputs = ("field_shift", operating_dimension, dimension_to)
        self._energy -= 1
        self.param_count = 1

    def add_projection(self, operating_dimension, dimension_to, projection_key):
        if self.golden_lock:
            raise Exception("Invalid operation on golden cell")
        self.left_index = operating_dimension
        self.right_index = -1
        self.generation_state = projection_key
        self.return_index = dimension_to
        self.action_type = ActionSet.PROJECTION
        self.graph_op_inputs = ("projection", operating_dimension, dimension_to, projection_key)
        self._energy -= 3
        self.param_count = 1

    def add_softmax(self, operating_dimension):
        if self.golden_lock:
            raise Exception("Invalid operation on golden cell")
        self.left_index = operating_dimension
        self.right_index = -1
        self.action_type = ActionSet.SOFTMAX
        self.return_index = operating_dimension
        self.graph_op_inputs = ("softmax", operating_dimension)
        self._energy -= 5
        self.param_count = 1

    def add_bell(self, operating_dimension):
        if self.golden_lock:
            raise Exception("Invalid operation on golden cell")
        self.left_index = operating_dimension
        self.action_type = ActionSet.BELL
        self.return_index = operating_dimension
        self.graph_op_inputs = ("bell", operating_dimension)
        self._energy -= 5
        self.param_count = 1

    def add_einsum(self, operating_dimension, dimension_with, dimension_to):
        if self.golden_lock:
            raise Exception("Invalid operation on golden cell")
        self.left_index = operating_dimension
        self.right_index = dimension_with
        self.return_index = dimension_to
        self.action_type = ActionSet.EINSUM
        self.graph_op_inputs = ("einsum", operating_dimension, dimension_with, dimension_to)
        self._energy -= 20
        self.param_count = 2
        self.right_index = dimension_with

    def add_conv(self, operating_dimension, dimension_to, conv_gen_state):
        if self.golden_lock:
            raise Exception("Invalid operation on golden cell")
        self.left_index = operating_dimension
        self.right_index = -1
        self.return_index = dimension_to
        self.action_type = ActionSet.CONV
        self.generation_state = conv_gen_state
        self.graph_op_inputs = ("conv", operating_dimension, dimension_to, conv_gen_state)
        self._energy -= 10
        self.param_count = 1

    def move(self, delta_w, delta_x, delta_y, delta_z):
        if self.golden_lock:
            raise Exception("Invalid operation on golden cell")
        self.w += delta_w
        self.x += delta_x
        self.y += delta_y
        self.z += delta_z
        self._energy -= math.ceil(math.sqrt(delta_x ** 2 + delta_y ** 2))

    def mate(self):
        if 100 < self._energy:
            self._energy -= 100
            return True
        else:
            return False

    def update_epigene(self, gene_layer, index, value):
        update_var = np.array(self.epigenetics[gene_layer])
        update_var[index] = value
        self._epigenetics[gene_layer].assign(update_var)

    def add_epigene(self, gene_layer, index):
        self.update_epigene(gene_layer, index, 0.0)
        self._energy -= 1

    def subtract_epigene(self, gene_layer, index_heat_map):
        normalized_index_heat_map = tf.math.sigmoid(index_heat_map) + .01
        filtered_index_heat_map = tf.multiply(normalized_index_heat_map, self.epigenetics[gene_layer])
        index = tf.argmax(filtered_index_heat_map)
        self.update_epigene(gene_layer, index, 1.0)
        self._energy -= 1

    def reset(self):
        if self.golden_lock:
            raise Exception("Invalid operation on golden cell")
        self.left_index = -1
        self.right_index = -1
        self.return_index = -1
        self.action_type = -1
        self.generation_state = np.zeros([OP_GEN_STATE_SIZE], dtype=np.float32)

    def lock(self):
        action_array = [0] * len(ActionSet)
        action_array[ActionSet.MOVE] = 1
        action_array[ActionSet.MATE] = 1
        action_array[ActionSet.ADD_EPIGENE] = 1
        action_array[ActionSet.REMOVE_EPIGENE] = 1
        action_array[ActionSet.UNLOCK] = 1
        action_array[ActionSet.TRANSFER_ENERGY] = 1
        action_array[ActionSet.WAIT] = 1
        self._action_mask = tf.constant(action_array, dtype=tf.float32)
        self._is_locked = True

    def unlock(self):
        mask = [1] * len(ActionSet)
        mask[ActionSet.UNLOCK] = 0
        self._action_mask = tf.constant(mask, dtype=tf.float32)
        self._is_locked = False

    def compute_transmit_amount(self, sender_key_and_hidden):
        hidden = tf.nn.relu(
            tf.matmul(tf.expand_dims(sender_key_and_hidden, axis=0), self.chromosomes[ChromosomeSet.RECEIVE_ENERGY][0]) +
            self.chromosomes[ChromosomeSet.RECEIVE_ENERGY][1]
        )
        energy_function_params = (tf.matmul(hidden, self.chromosomes[ChromosomeSet.RECEIVE_ENERGY][2]) +
            self.chromosomes[ChromosomeSet.RECEIVE_ENERGY][3])[0]
        energy_scale = tf.math.cos(energy_function_params[0] / energy_function_params[1])
        energy_transfer = energy_function_params[0] * energy_scale
        return energy_transfer

    def add_concat(self, operating_dimension, dimension_with, dimension_to):
        if self.golden_lock:
            raise Exception("Invalid operation on golden cell")
        self.left_index = operating_dimension
        self.right_index = dimension_with
        self.return_index = dimension_to
        self.action_type = ActionSet.CONCAT
        self.graph_op_inputs = ("concat", operating_dimension, dimension_with, dimension_to)
        self._energy -= 5
        self.param_count = 2

    def add_multiply(self, operating_dimension, dimension_with):
        if self.golden_lock:
            raise Exception("Invalid operation on golden cell")
        self.left_index = operating_dimension
        self.right_index = dimension_with
        self.action_type = ActionSet.MULTIPLY
        self.return_index = operating_dimension
        self.graph_op_inputs = ("multiply", operating_dimension, dimension_with)
        self._energy -= 10
        self.param_count = 2

    def add_add(self, operating_dimension, dimension_with):
        if self.golden_lock:
            raise Exception("Invalid operation on golden cell")
        self.left_index = operating_dimension
        self.right_index = dimension_with
        self.action_type = ActionSet.ADD
        self.return_index = operating_dimension
        self.graph_op_inputs = ("add", operating_dimension, dimension_with)
        self._energy -= 10
        self.param_count = 2

    def provide_signal_input(self):

        reward_component = np.array([self.energy, self.last_reward, self.last_transmit_reward, self.last_receive_reward], dtype=np.float32)
        return np.concatenate([
            self.op_data,
            reward_component,
            self.position
        ])

    def create_node_export_data(self):
        return cell_data_to_array(self.action_type,
                                  self.left_index,
                                  self.right_index,
                                  self.return_index,
                                  self.generation_state,
                                  self.position)
