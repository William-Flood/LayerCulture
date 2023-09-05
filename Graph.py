import time

from Cell import Cell
from typing import Tuple, List
from Node import Node
from Field import Field
import tensorflow as tf
import numpy as np
import random
import time

class NodeSource:
    def __init__(self, cell, id):
        self.cell = cell
        self.id = id
        self.node = None
        self.reward_variable = 1.0

class Graph:
    def __init__(self, cells: List[Cell], fields: Tuple[Field], max_field_edge_count, starting_hot_count,
                 distance_scalings):
        self.sources = [NodeSource(cell, cell_index) for cell, cell_index in
                        zip(cells, range(len(cells)))]
        candidate_sources = [source for source in self.sources if source.cell.has_graph]
        candidate_source_scores = np.array(tf.reduce_sum(tf.gather(distance_scalings, [source.id for source in candidate_sources]), axis=-1))
        self.hot_count = starting_hot_count
        self.hot_sources = []
        while 0 < len(candidate_sources) and starting_hot_count > len(self.hot_sources):
            chosen_source = random.choices(candidate_sources, weights=candidate_source_scores, k=1)[0]
            chosen_index = candidate_sources.index(chosen_source)
            del candidate_sources[chosen_index]
            candidate_source_scores = np.delete(candidate_source_scores, chosen_index)
            self.hot_sources.append(chosen_source)
        self.fields = fields
        self.max_field_edge_count = max_field_edge_count
        self.start_nodes, self.nodes, self.output_nodes = self.generate_graph(max_field_edge_count, distance_scalings)
        self.op = None
        self.compile()
        self.evaluation_score = 0

    @property
    def node_count(self):
        return len(self.nodes)


    def generate_graph(self, max_field_edge_count, distance_scalings):
        # locked_length = len(tuple(ordering_metadata for ordering_metadata in cell_orderings if ordering_metadata[2]))
        distance_scalings_with_start = tf.concat(
            [
                tf.expand_dims(
                    tf.sqrt(tf.reduce_sum(tf.square(
                        [
                            [
                            source.cell.w,source.cell.x,source.cell.y,source.cell.z
                            ] for source in self.sources
                        ]
                        ), axis=1)
                    ), axis=1),
                distance_scalings
            ],
            axis=1
        )
        self.sources.sort(key=lambda s: s.cell.z)
        field_nodes = tuple([] for _ in self.fields)
        total_nodes = []

        sector_inputs = dict()
        def entry_op(field_index):
            return sector_inputs[field_index]

        def make_start_node(field):
            return Node(
                node_index=field.field_index,
                candidate_edges=field_nodes,
                node_op_provider=lambda:
                    lambda node_op_states: entry_op(field.field_index),
                w=0, x=0, y=0, z=0,
                field_visited_check=lambda field_index: field_index == field.field_index,
                field=self.fields[field.field_index],
                op_providers=f"Sector entry {field.field_index}",
                max_field_edge_count=max_field_edge_count,
                distance_scalings=distance_scalings
            )

        start_nodes = []
        start_node = make_start_node(self.fields[0])
        field_nodes[0].append(start_node)
        # total_nodes.append(start_node)
        start_nodes.append(start_node)

        construction_start = time.time()
        node_times = []
        for source in self.hot_sources:
            node_create_start = time.time()
            cell = source.cell
            # Cells contain a list of Tensorflow ops, which act on aggregated values emmited to the assembly of fields in the
            # environment.  The last op emits a new value into the environment.  cell.generate_ops is a function that returns
            # a function to loop through those ops.
            cell_op_provider = cell.generate_ops
            cell_node = Node(
                source.id + 1,
                field_nodes,
                cell_op_provider,
                cell.w, cell.x, cell.y, cell.z,
                cell.visited,
                self.fields[cell.field_index],
                cell,
                max_field_edge_count,
                distance_scalings_with_start[source.id]
            )
            source.node = cell_node
            field_nodes[cell.field_index].append(cell_node)
            node_create_time = time.time() - node_create_start
            node_times.append(node_create_time)
            total_nodes.append(cell_node)
        # Filter out nodes that don't contribute to the final result of the graph execution
        # Start by setting the node usage to true or false, based on whether it outputs to the global output field, and
        # returning the nodes with edges that point to the nodes that are marked as used

        output_nodes = [node for node in total_nodes if node.output_field == self.fields[-1] and node.is_valid]
        if 0 < len(output_nodes):
            usage_updates = [traversed_node
                             for output_node in output_nodes
                                for traversed_node in output_node.reset_usage()
                             ]
            while 0 < len(usage_updates):
                # For each node that had been marked as used in the last usage determination pass, mark the nodes
                # connected to the edges pointing to that node as used, and return the nodes with edges that point to the
                # nodes that are marked as used
                usage_updates = [marked_node for last_marked_node in usage_updates for marked_node in last_marked_node.mark_used_edges()]
            for node in start_nodes:
                node.reset_usage()
            used_nodes = [node for node in total_nodes if node.is_used]
        else:
            used_nodes = []
        construction_time = time.time() - construction_start
        self.hot_sources = [source for source in self.hot_sources if source.node.is_used]
        return start_nodes, used_nodes, output_nodes

    def compile(self):
        node_ops_and_vars = [node.make_node_eval(
            test_mode=True,
            fields=self.fields,
        ) for node in self.nodes]
        def graph_op(input_val, batch_size):
            self.start_nodes[0].emitted_value = input_val
            for start_node, field in zip(self.start_nodes[1:], self.fields[1:]):
                start_node.emitted_value = tf.zeros([batch_size] + field.shape)
            for node_op in node_ops_and_vars:
                node_op()
            return tf.reduce_sum(tf.stack([output_node.emitted_value for output_node in self.output_nodes]), axis=0)
        self.op = graph_op



    def operate(self, input_val):
        batch_size = int(tf.shape(input_val)[0])
        return self.op(input_val, batch_size)

    def train(self, training_iter, trainer, training_cycles):
        training_vars = [source.node.training_var for source in self.hot_sources]
        losses = []
        cycle_batch = next(training_iter)
        input_val = cycle_batch["inputs"]
        output_val = cycle_batch["outputs"]
        batch_size = int(tf.shape(input_val)[0])
        eval_start = time.time()
        unscaled_return = self.op(input_val, batch_size)
        eval_time = time.time() - eval_start
        return_sum = tf.reduce_sum(unscaled_return)
        if(float(return_sum) != 0):
            return_scaling = tf.reduce_sum(output_val) / return_sum
        else:
            return_scaling = 1.0
        for step in range(training_cycles):
            with tf.GradientTape() as graph_tape:
                return_val = self.op(input_val, batch_size) * return_scaling
                loss = tf.reduce_sum(tf.square(return_val - output_val))
                losses.append(loss)
            grads = graph_tape.gradient(loss, training_vars)
            trainer.apply_gradients(grads_and_vars=zip(grads, [source.node.training_var for source in self.hot_sources]))
        for source in self.hot_sources:
            source.reward_variable = source.node.training_var
        self.evaluation_score = loss * (1 + eval_time)


    def copy(self, copy_amount, scale_start, scale_end, distance_scalings):
        for copy_index in range(copy_amount):
            count_scaling = int(self.hot_count * (scale_end * (copy_index / (copy_amount - 1.0)) +
                                                  scale_start * (1.0 - (copy_index / (copy_amount - 1.0)))))
            yield Graph([source.cell for source in self.sources], self.fields, self.max_field_edge_count,
                        count_scaling, distance_scalings)

    def allocate_energy(self, energy, distance_scalings):
        if 0 < len(self.hot_sources):
            live_selectors = [source.id for source in self.sources if not (source.cell.has_graph or source.cell.is_dead)]
            selector_cell_scalings = tf.gather(distance_scalings, live_selectors, axis=1)
            live_hot_sources = [source for source in self.hot_sources if not source.cell.is_dead]
            tangent_reward_scalings = tf.nn.softmax(tf.gather(selector_cell_scalings, [source.id for source in live_hot_sources]), axis=1)
            node_reward_scalings = tf.nn.softmax(tf.stack([source.reward_variable for source in live_hot_sources]))


            for source, node_reward_scaling in zip(live_hot_sources, node_reward_scalings):
                if not source.cell.is_dead:
                    source.cell.reward(energy * node_reward_scaling)

            node_reward_scalings_ext = tf.expand_dims(node_reward_scalings, 1)
            reward_distance_scalings = tf.multiply(node_reward_scalings_ext, tangent_reward_scalings)
            tangent_rewards = tf.reduce_sum(reward_distance_scalings, axis=0)
            for live_selector_index, tangent_reward in zip(live_selectors, tangent_rewards):
                self.sources[live_selector_index].cell.reward(energy * live_selector_index)

    def refresh_sources(self, indices, cells):
        for cell_index in indices:
            self.sources[cell_index] = NodeSource(cells[cell_index], cell_index)

    def serialize(self):
        return "[" + ", ".join([node.serialize() for node in self.hot_sources]) + "]"
