import time
import tensorflow as tf
from typing import Callable
from Field import Field
import numpy as np


# This function is used by a node to create an 'assembly' for a field,
# which is a function that when called will gather values emitted into this field by this node's input edges.
def make_field_assembly(field_edges, field,
                        distance_scaling
                        ):
    def add_field_values(cell_op_state):
        emitted_inputs = tf.stack([edge.emitted_value for edge in field_edges])
        clipped_scale_value = field.gather_field_graph_values(emitted_inputs, distance_scaling)
        cell_op_state[field.field_index] = clipped_scale_value

    # assemble_inputs.append(add_field_values)
    return add_field_values





class Node:
    def __init__(self, node_index, candidate_edges, node_op_provider: Callable, w, x, y, z,
                 field_visited_check: Callable, field: Field, op_providers,
                 max_field_edge_count, distance_scalings):
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
             max_field_edge_count (int): The maximum number of edges, per field that this node can be assigned
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
        self.edges = []
        self._output_field = field
        self._used = False
        # A record of the edges connected to this node by index - a subset of these will be used in tensor gather operations

        self.field_edge_length_scalings = []
        for field_index, field_edges in enumerate(candidate_edges):
            valid_field_edges = [edge for edge in field_edges if edge.is_valid]
            if 0 < len(valid_field_edges) and field_visited_check(field_index):
                field_edge_selection_and_distances = self.size_and_filter_edges(
                    distance_scalings,
                    max_field_edge_count,
                    valid_field_edges
                )
                self.edges.append([field_edge_selection_and_distance[0]
                                   for field_edge_selection_and_distance in field_edge_selection_and_distances])
                self.field_edge_length_scalings.append([field_edge_selection_and_distance[1]
                                   for field_edge_selection_and_distance in field_edge_selection_and_distances])
            else:
                self.edges.append([])
                self.field_edge_length_scalings.append([])


        self._is_valid = all(
            0 < len(field_edges) or field_index == 0 for field_index, field_edges in enumerate(self.edges) if
            field_visited_check(field_index))
        self._field_visited_check = field_visited_check
        self.op_providers = op_providers
        self.emitted_value = None
        self.training_var = tf.Variable(1.0)

    @property
    def field_edge_counts(self):
        return tuple(len(field_edges) for field_edges in self.edges)

    @property
    def is_valid(self):
        return self._is_valid

    @property
    def is_used(self):
        return self._used

    def size_and_filter_edges(self, distance_scalings, max_edge_count, candidate_edges):
        edge_and_scaling = [[edge, distance_scalings[edge.index]] for edge in candidate_edges]
        edge_and_scaling.sort(key=lambda e: e[1])
        seleted_edges_and_distances = edge_and_scaling[-1 * max_edge_count:]
        return seleted_edges_and_distances


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
        flattened_edges = [edge for field_edge in self.edges for edge in field_edge]
        return flattened_edges

    @property
    def output_field(self):
        return self._output_field

    def reset_usage(self, output_index=None):
        """
        Resets the usage status of the node based on its output field
        :param output_index: The index of the field output by the graph
        :return: The edges in this node's computation if it marked as used, or else an empty tuple
        """
        self._used = (output_index is None or output_index == self._output_field.field_index)
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

    def make_node_eval(self, test_mode, fields,):
        """
        Creates a function for evaluating the node.

        Args:
            test_mode (bool): True if the evaluation is in test mode, False otherwise.
            fields (List[Field]): The list of fields.

        Returns:
            Tuple: The computation performed by this node, and a list of trainable variables used to give feedback to the ecosystem.
        """
        assemble_inputs = []
        # noinspection PyCallingNonCallable
        node_op = self.node_op_provider()
        for field_edges, field, edge_length_scalings in zip(self.edges, fields, self.field_edge_length_scalings):
            edge_count = len(field_edges)
            if 0 < edge_count:
                op = make_field_assembly(field_edges, field,
                                         edge_length_scalings)
                assemble_inputs.append(op)


        if test_mode:
            # @tf.function
            def emit_node_value():
                node_op_states = dict()
                for assemble_op in assemble_inputs:
                    assemble_op(node_op_states)
                # noinspection PyCallingNonCallable
                op_return = node_op(node_op_states) * self.training_var
                tf.debugging.check_numerics(op_return, "Node result had invalid data")
                self.emitted_value = op_return
        else:
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

    def serialize(self):
        if 0 == len(self.edges):
            node_op = self.node_op_provider
        else:
            node_op = self.node_op_provider.graph_op_inputs

        return (f"{{" +
                ", ".join([
                    f"id: {self.index}",
                    f"edges: {[[field_index, [edge.index for edge in edge_array]] for field_index, edge_array in enumerate(self.edges)]}",
                    f"nodeOp: {node_op}"
                ]) +
                f"}}")
