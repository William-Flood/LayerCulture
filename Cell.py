import tensorflow as tf
import numpy as np
import math
from typing import Tuple
from Field import Field
from CellEnums import ActionSet, ChromosomeSet


def breed_cells(mother_genome, father_genome, mutation_rate):
    chromosomes = []
    for mother_chromosome, father_chromosome in zip(mother_genome, father_genome):
        chromosome = []
        for mother_gene, father_gene in zip(mother_chromosome, father_chromosome):
            mutation = tf.random.normal(tf.shape(mother_gene), mean=0.0, stddev=mutation_rate)
            chromosome.append(
                (mother_gene + father_gene) / 2 + mutation
            )
        chromosomes.append(chromosome)


STARTING_MASK = tf.constant([1] + [0] * (len(ActionSet) - 1), dtype=tf.float32)

class Cell:
    def __init__(self, fields : Tuple[Field], chromosomes, epigenetics, w, x, y, z):
        pass
        self.fields = fields
        self._field_index = -1
        self.graph_ops = []
        self.chromosomes = chromosomes
        self._epigenetics = epigenetics
        self._energy = 100.0
        self._action_mask = STARTING_MASK
        self._gradient_acceptor = tf.Variable(1.0)
        self.w = w
        self.x = x
        self.y = y
        self.z = z
        self._last_reward = 0
        self._transmit_reward = 0
        self._receive_reward = 0
        self._visited_fields = set()
        self._is_locked = False
        self.graph_op_inputs = []

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

    def conclude_transmit(self, energy):
        self._energy += energy
        self._transmit_reward = energy

    @property
    def last_transmit_reward(self):
        return self._transmit_reward

    @property
    def last_receive_reward(self):
        return self._receive_reward

    @property
    def is_dead(self):
        return self._energy <= 0

    @property
    def current_output_field(self):
        return self.fields[self._field_index]

    @property
    def visited_field(self):
        return tuple(self._visited_fields)

    @property
    def is_locked(self):
        return self._is_locked

    def visited(self, field_index):
        return field_index in self._visited_fields

    @property
    def has_graph(self):
        return 0 < len(self.graph_ops)

    def accept_environment_epigene_manifext(self, manifest):
        self.epigene_manifest = manifest

    def add_field_shift(self, dimension_to):
        self._visited_fields.add(self._field_index)
        self.graph_ops.append(self.current_output_field.dimensional_shift[dimension_to])
        self.graph_op_inputs.append(("field_shift", self._field_index, dimension_to))
        self._field_index = dimension_to
        self._energy -= 1

    def add_projection(self, dimension_to, projection_key):
        self._visited_fields.add(self._field_index)
        self.graph_ops.append(self.current_output_field.add_projection[dimension_to](projection_key))
        self.graph_op_inputs.append(("projection", self._field_index, dimension_to, projection_key))
        self._field_index = dimension_to
        self._energy -= 3

    def add_softmax(self):
        self.graph_ops.append(self.current_output_field.softmax)
        self.graph_op_inputs.append(("softmax", self._field_index))
        self._energy -= 5

    def add_bell(self):
        self.graph_ops.append(self.current_output_field.bell)
        self.graph_op_inputs.append(("bell", self._field_index))
        self._energy -= 5

    def add_einsum(self, dimension_with, dimension_to):
        self._visited_fields.add(self._field_index)
        self.graph_ops.append(self.current_output_field.build_einsum_op(self.fields[int(dimension_with)], self.fields[int(dimension_to)]))
        self.graph_op_inputs.append(("einsum", self._field_index, dimension_with, dimension_to))
        self._visited_fields.add(dimension_with)
        self._field_index = dimension_to
        self._energy -= 20

    def add_conv(self, dimension_to, conv_gen_state):
        self._visited_fields.add(self._field_index)
        self.graph_ops.append(self.current_output_field.make_conv[dimension_to](conv_gen_state))
        self.graph_op_inputs.append(("conv", self._field_index, dimension_to, conv_gen_state))
        self._field_index = dimension_to
        self._energy -= 10

    def move(self, delta_w, delta_x, delta_y, delta_z):
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

    def set_golden(self):
        action_array = [0] * len(ActionSet)
        action_array[ActionSet.MATE] = 1
        action_array[ActionSet.ADD_EPIGENE] = 1
        action_array[ActionSet.REMOVE_EPIGENE] = 1
        action_array[ActionSet.TRANSFER_ENERGY] = 1
        action_array[ActionSet.WAIT] = 1
        self._action_mask = tf.constant(action_array, dtype=tf.float32)
        self._is_locked = True


    def accept_energy(self, sender_key_and_hidden):
        hidden = tf.nn.relu(
            tf.matmul(sender_key_and_hidden, self.chromosomes[ChromosomeSet.RECEIVE_ENERGY][0]) +
            self.chromosomes[ChromosomeSet.RECEIVE_ENERGY][1]
        )
        energy_function_params = tf.matmul(hidden, self.chromosomes[ChromosomeSet.RECEIVE_ENERGY][2]) +\
            self.chromosomes[ChromosomeSet.RECEIVE_ENERGY][3]
        energy_scale = tf.math.cos(energy_function_params[0] / energy_function_params[1])
        energy_transfer = energy_function_params[0] * energy_scale
        if energy_transfer > self._energy:
            energy_transfer = self._energy
        self._energy -= energy_transfer
        self._receive_reward -= energy_transfer
        return energy_transfer

    def reset(self, starting_field):
        self._field_index = starting_field
        self.graph_ops = []
        self.graph_op_inputs = [("reset", starting_field)]
        self.unlock()
        self._visited_fields = set()
        self._visited_fields.add(starting_field)
        pass

    def add_concat(self, dimension_with, dimension_to):
        self._visited_fields.add(self._field_index)
        concat_op = self.current_output_field.make_concat(dimension_with, dimension_to)
        self.graph_ops.append(concat_op)
        self.graph_op_inputs.append(("concat", self._field_index, dimension_with, dimension_to))
        self._field_index = dimension_to
        self._visited_fields.add(dimension_with)
        self._energy -= 5

    def add_multiply(self, dimension_with):
        multiply_op = self.current_output_field.make_multiply[dimension_with]
        self.graph_ops.append(multiply_op)
        self.graph_op_inputs.append(("multiply", self._field_index, dimension_with))
        self._visited_fields.add(dimension_with)
        self._energy -= 10

    def add_add(self, dimension_with):
        add_op = self.current_output_field.make_add[dimension_with]
        self.graph_ops.append(add_op)
        self.graph_op_inputs.append(("add", self._field_index, dimension_with))
        self._visited_fields.add(dimension_with)
        self._energy -= 10

    def generate_ops(self):
        # @tf.function
        def ops(cell_op_state):
            instance_state = cell_op_state.copy()
            for op in self.graph_ops:
                op(instance_state)
            clipped_result = tf.clip_by_value(
                instance_state[self._field_index],
                clip_value_min=-1e4,
                clip_value_max=1e4
            )
            return clipped_result
        return ops
