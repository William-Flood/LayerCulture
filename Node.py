import time

import tensorflow as tf
from typing import Callable
from Field import Field
import numpy as np


# This function is used by a node to create an 'assembly' for a field,
# which is a function that when called will gather values emitted into this field by this node's input edges.
def make_field_assembly(field_edges, field_edge_selection, field,
                        fields_emission_positions_lists, node_position, test_mode
                        ):
    edge_count = len(field_edges)
    if 0 < edge_count:
        selection_count = len(field_edge_selection)
        field_emission_positions = tf.gather(fields_emission_positions_lists, field_edge_selection)
        field_emission_displacements = field_emission_positions - tf.tile(tf.reshape(node_position, [1, 4]),
                                                                          [edge_count, 1])
        # Determines the distance of each emitted value in the field from the target location
        distances = tf.math.sqrt(tf.reduce_sum(tf.square(field_emission_displacements), axis=1))
        node_ones = tf.ones([selection_count], dtype=tf.float32)
        # Scaling factor to ensure that each emitted value for the field contributes the full evaluated value at the source, and gradually decays with distance
        distance_scaling = tf.divide(node_ones, distances + node_ones)
        if test_mode:
            field_test_gradients = tf.Variable(distance_scaling)

            def test_add_field_values(cell_op_state):
                emitted_inputs = tf.stack([edge.emitted_value for edge in field_edges])
                clipped_scale_value = field.gather_field_graph_values(emitted_inputs, field_test_gradients)

                clipped_scale_value = tf.clip_by_value(clipped_scale_value,
                                                       clip_value_min=-1e4,
                                                       clip_value_max=1e4
                                                       )
                cell_op_state[field.field_index] = clipped_scale_value
            return test_add_field_values, field_test_gradients
        else:
            def add_field_values(cell_op_state):
                emitted_inputs = tf.stack([edge.emitted_value for edge in field_edges])
                clipped_scale_value = field.gather_field_graph_values.gather_op(emitted_inputs, distance_scaling)
                cell_op_state[field.field_index] = clipped_scale_value

            # assemble_inputs.append(add_field_values)
            return add_field_values
    else:
        zeros_size = [1] + field.shape
        def add_field_values(cell_op_state):
            scaled_value = tf.zeros(zeros_size)
            tf.debugging.check_numerics(scaled_value, "Scaling evaluated to an invalid value")
            cell_op_state[field.field_index] = scaled_value

        return add_field_values


class Node:
    def __init__(self, node_index, candidate_edges, node_op_provider: Callable, w, x, y, z,
                 field_visited_check: Callable, field: Field,
                 op_providers):
        """
         Represents a node in the graph.

         Args:
             node_index (int): The index of the node.
             candidate_edges (List[List[int]]): All nodes that this node could potentially connect to, sorted into the candidate nodes' output fields.
             node_op_provider (Callable): A function that provides the node's operation.
             w (float): The w-coordinate of the node's position in the ecosystem.
             x (float): The x-coordinate of the node's position in the ecosystem.
             y (float): The y-coordinate of the node's position in the ecosystem.
             z (float): The z-coordinate of the node's position in the ecosystem.
             field_visited_check (Callable): A function to check if a field is in use by this node's computation
             field (Field): The output field associated with the node.
         """
        self.index = node_index
        # w, x, y, and z represent the position of the node in the ecosystem; used to scale the node's emitted value as
        # its connected nodes consume that value in their own calculations
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.node_op_provider = node_op_provider
        # Filters the candidate edges according to whether this node's computation relies on the value emitted from the considered edge
        self.edges = tuple(
            [edge for edge in field_edges] if field_visited_check(field_index) else tuple() for field_index, field_edges
            in enumerate(candidate_edges))
        # Unused - if a call to this node's operation is made without a connection available to one of its input fields, a zero-valued tensor will be
        # used instead
        self._is_valid = all(
            0 < len(field_edges) or field_index == 0 for field_index, field_edges in enumerate(candidate_edges) if
            field_visited_check(field_index))
        self._output_field = field
        self._used = False
        # A record of the edges connected to this node by index - a subset of these will be used in tensor gather operations
        self.full_edge_selection = tuple(tuple(edge_index for edge_index, _ in enumerate(field_edges)) for field_edges in self.edges)
        # The subset of edges used in this node's operation
        self.edge_selection = self.full_edge_selection
        self._field_visited_check = field_visited_check
        self.op_providers = op_providers
        self.emitted_value = None
        self.test_gradients = []

    @property
    def field_edge_counts(self):
        return tuple(len(field_edges) for field_edges in self.edges)

    @property
    def is_valid(self):
        return self._is_valid

    @property
    def is_used(self):
        return self._used

    def mark_used_edges(self):
        """
        Marks the node as used, and returns
        :return: The edges in this node's computation if it was previously not marked as used, or an empty tuple if it had been previously marked
        had already been marked as used
        """
        if self._used:
            return tuple()
        else:
            self._used = True
            return tuple(edge for edge in self.flatten_edges())

    def flatten_edges(self):
        for field_edges in self.edges:
            for edge in field_edges:
                yield edge

    @property
    def output_field(self):
        return self._output_field

    def reset_usage(self, output_index):
        """
        Resets the usage status of the node based on its output field
        :param output_index: The index of the field output by the graph
        :return: The edges in this node's computation if it marked as used, or else an empty tuple
        """
        self._used = (output_index == self._output_field.field_index)
        if self._used:
            return tuple(edge for edge in self.flatten_edges())
        else:
            return tuple()

    def visited_field(self, field_index):
        """
        Provides a check if this node's computation relies on a given field
        :param field_index: The index of the field to check
        :return: The result of the check
        """
        return self._field_visited_check(field_index)

    def make_node_eval(self, test_mode, fields, fields_emission_positions_lists):
        """
        Creates a function for evaluating the node.

        Args:
            test_mode (bool): True if the evaluation is in test mode, False otherwise.
            fields (List[Field]): The list of fields.
            fields_emission_positions_lists (List[List[tf.Tensor]]): The emission positions of the fields.

        Returns:
            Tuple: The computation performed by this node, and a list of trainable variables used to give feedback to the ecosystem.
        """
        assemble_inputs = []
        node_position = tf.constant([self.w, self.x, self.y, self.z])
        # noinspection PyCallingNonCallable
        node_op = self.node_op_provider()
        for field_edges, field_edge_selection, field in zip(self.edges, self.edge_selection, fields):
            edge_count = len(field_edges)
            if 0 < edge_count:
                if test_mode:
                    op, vars = make_field_assembly(field_edges, field_edge_selection, field,
                                                   fields_emission_positions_lists, node_position, test_mode)
                    assemble_inputs.append(op)
                    self.test_gradients.append(vars)
                else:
                    op = make_field_assembly(field_edges, field_edge_selection, field, fields_emission_positions_lists,
                                             node_position, test_mode)
                    assemble_inputs.append(op)
            elif self.visited_field(field.field_index):
                op = make_field_assembly(field_edges, field_edge_selection, field, fields_emission_positions_lists,
                                         node_position, test_mode)
                assemble_inputs.append(op)


        # @tf.function
        def emit_node_value():
            node_op_states = dict()
            for assemble_op in assemble_inputs:
                assemble_op(node_op_states)
            # noinspection PyCallingNonCallable
            op_return = node_op(node_op_states)
            tf.debugging.check_numerics(op_return, "Node result had invalid data")
            self.emitted_value = op_return

        return emit_node_value
