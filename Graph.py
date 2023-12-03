import time
from numpy.typing import NDArray
from typing import Tuple
from Node import Node
from Field import Field
import tensorflow as tf
import numpy as np
import random
import time
import pickle
from cell_export_manager import start_data, array_to_cell_data

class NodeSource:
    def __init__(self, cell_export_data, id):
        self.cell_export_data = cell_export_data
        self.id = id
        self.node = None
        self.reward_variable = 0.0
        self.action_type, self.left_index, self.right_index, self.return_index, self.generation_state, self.position = \
            array_to_cell_data(cell_export_data)

    @property
    def has_graph(self):
        return self.action_type is not None

    @property
    def z(self):
        return self.position[-1]


class Graph:
    def __init__(self, cells_export_data: NDArray, fields: Tuple[Field], max_field_edge_count, starting_hot_count,
                 distance_scalings, last_golden_indices):
        self.log = dict()
        self.sources = [NodeSource(cell, cell_index) for cell_index, cell in
                        enumerate(cells_export_data)]
        last_golden = [source for source, was_golden in zip(self.sources, last_golden_indices) if source and was_golden]
        if any([source.cell_export_data[0] == -1 for source in last_golden]):
            raise Exception("Golden cell updated to non-operation")
        candidate_sources = [source for source, was_golden in zip(self.sources, last_golden_indices) if source.has_graph and not was_golden]
        candidate_source_scores = np.array(tf.reduce_sum(tf.gather(distance_scalings, [source.id for source in candidate_sources]), axis=-1))
        candidate_source_probabilities = candidate_source_scores / np.sum(candidate_source_scores)
        self.log["candidate_source_scores"] = candidate_source_scores
        self.hot_count = min(starting_hot_count, len(candidate_sources))
        self.hot_sources = [candidate_sources[i] for i in
                            np.random.choice(
                                len(candidate_sources),
                                size=self.hot_count - len(last_golden),
                                replace=False,
                                p=candidate_source_probabilities
                            )
                            ] + last_golden
        self.fields = fields
        self.max_field_edge_count = max_field_edge_count
        self.start_nodes, self.nodes, self.output_nodes = self.generate_graph(max_field_edge_count, distance_scalings)
        self.total_nodes = self.nodes
        self.log["graph_serialization"] = self.serialize()
        self.op, self.training_variables = self.compile()
        self.evaluation_score = 0
        self.end_loss = None

    @property
    def node_count(self):
        return len(self.nodes)


    def generate_graph(self, max_field_edge_count, distance_scalings):
        """
        Generates a graph structure based on a set of source cells and distance scalings.

        This method initializes the graph structure by creating nodes from the selected "hot" cells of the graph. Each node represents a cell's
        operation within a given field in the ecosystem. The generated nodes are then interconnected based on the distance
        scalings, producing a series of computations ordered based on the source cells' z position. The start nodes serve as the entry points to each field,
        while the overall structure ensures that the flow of operations culminates at the output nodes.

        Args:
            max_field_edge_count (int): The maximum number of edges a field node can have.
            distance_scalings (tf.Tensor): A tensor containing scalings based on distances between cells. It plays a
                                           crucial role in determining the connectivity of the nodes in the graph.

        Returns:
            tuple: A tuple containing:
                - start_nodes (list[Node]): A list of nodes that serve as the entry points to the various fields in the ecosystem.
                - nodes (list[Node]): A list of nodes representing the hot sources and their operations within the fields.
                - output_nodes (list[Node]): A list of nodes that serve as the termination points of the graph, representing
                                             outputs to the global output field.

        Note:
            - Nodes that do not contribute to the final result of the graph execution are filtered out.
        """
        distance_scalings_with_start = tf.concat(
            [
                tf.expand_dims(
                    tf.sqrt(tf.reduce_sum(tf.square(
                        [
                            source.position for source in self.sources
                        ]
                        ), axis=1)
                    ), axis=1),
                distance_scalings
            ],
            axis=1
        )
        self.hot_sources.sort(key=lambda s: s.z)
        field_nodes = tuple([] for _ in self.fields)
        total_nodes = []

        sector_inputs = dict()
        def entry_op(field_index):
            return sector_inputs[field_index]

        def make_start_node(field):
            return Node(
                node_index=field.field_index,
                candidate_edges=field_nodes,
                generation_data=start_data(),
                field=self.fields[field.field_index],
                max_field_edge_count=max_field_edge_count,
                distance_scalings=distance_scalings,
                is_start=True
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
            cell_node = Node(
                source.id + 1,
                field_nodes,
                source.cell_export_data,
                self.fields[source.return_index],
                max_field_edge_count,
                distance_scalings_with_start[source.id]
            )
            source.node = cell_node
            field_nodes[source.return_index].append(cell_node)
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
        """
        Compiles the Graph into a callable operation for TensorFlow execution.  The method returns a callable operation (`graph_op`) that accepts 
        an input tensor.  This input is run through a convolutional layer, and then operated on by the Graph's nodes.  The output of the graph is then 
        run through a fully connected layer to generate the final output.

        Returns:
            graph_op (callable): A compiled TensorFlow operation that represents the computation encapsulated
                                 by the Graph. When called, it accepts an input tensor and returns the result of 
                                 processing this tensor through the Graph.
            start_end_layers (List[tf.Variable]): A list of TensorFlow variables representing the parameters 
                                                   of the input and output layers of the Graph. These include 
                                                   weights and biases for both layers.
        """
        edge_training_var = tf.Variable(tf.ones([len(self.nodes), len(self.fields), self.max_field_edge_count, 1]), name="edges")
        node_ops_and_vars = [node.make_node_eval(
            test_mode=True,
            fields=self.fields,
            training_variable=edge_training_var,
            hot_node_id=hot_node_id
        ) for node, hot_node_id in zip(self.nodes, range(len(self.nodes))) if node.is_active]
        start_layer_size = self.fields[0].shape[-1]
        start_w = tf.Variable(tf.random.normal([5, 5, start_layer_size, start_layer_size]), name="start_w")
        start_b = tf.Variable(tf.Variable(tf.zeros([start_layer_size])), name="start_b")

        end_size = self.fields[-1].shape[0]
        end_w = tf.Variable(tf.random.normal([end_size, end_size]), name="end_w")
        end_b = tf.Variable(tf.zeros([end_size]), name="end_b")
        # @tf.function
        def graph_op(input_val):
            prod = tf.nn.conv2d(input_val, start_w, strides=[1, 1], padding="SAME")
            prod_and_bias = prod + start_b

            self.start_nodes[0].emitted_value = prod_and_bias
            for node_op in node_ops_and_vars:
                node_op()
            graph_end = tf.reduce_sum(tf.stack([output_node.emitted_value for output_node in self.output_nodes]), axis=0)
            return tf.matmul(graph_end, end_w) + end_b
        return graph_op, [start_w, start_b, end_w, end_b, edge_training_var]



    def operate(self, input_val):
        batch_size = int(tf.shape(input_val)[0])
        return self.op(input_val, batch_size)

    def train_variables(self, training_cycles, training_iter, unscaled_return_mu, output_return_sigma, unscaled_return_sigma, output_return_mu, trainer):
        losses = []
        gradients = []
        times = []
        for step in range(training_cycles):
            step_start = time.time()
            cycle_batch = next(training_iter)
            input_val = cycle_batch["inputs"]
            output_val = cycle_batch["outputs"]
            with tf.GradientTape() as graph_tape:
                return_val = (self.op(input_val) - unscaled_return_mu) * (output_return_sigma / unscaled_return_sigma) + output_return_mu
                loss = tf.reduce_sum(tf.square(return_val - output_val))
                losses.append(loss)
            grads = graph_tape.gradient(loss, self.training_variables)
            gradients.append(grads)
            trainer.apply_gradients(grads_and_vars=zip(grads, self.training_variables))
            times.append(time.time() - step_start)
        return losses, gradients, times


    def prune_nodes(self, training_iter, unscaled_return_mu, output_return_sigma, unscaled_return_sigma, output_return_mu, sample_amount, remove_amount):
        active_nodes = [node for node in self.nodes if node.is_active]
        sample_count = int(len(active_nodes) * sample_amount)
        removal_candidates = random.sample(active_nodes, sample_count)
        for node in removal_candidates:
            node.training_variable.assign(.9)
        training_vars = [node.training_variable for node in removal_candidates]
        cycle_batch = next(training_iter)
        input_val = cycle_batch["inputs"]
        output_val = cycle_batch["outputs"]
        with tf.GradientTape() as graph_tape:
            return_val = (self.op(input_val) - unscaled_return_mu) * (output_return_sigma / unscaled_return_sigma) + output_return_mu
            loss = tf.reduce_sum(tf.square(return_val - output_val))
        grads = graph_tape.gradient(loss, training_vars)
        enumerated_grads = [index_and_grad for index_and_grad in enumerate(grads) if index_and_grad[1] is not None]
        enumerated_grads.sort(key=lambda eg: eg[1])
        remove_past_index = int(sample_count * (1 - remove_amount))
        removed_nodes = [removal_candidates[removal_index] for removal_index, _ in enumerated_grads[remove_past_index:]] \
            + [removal_candidates[removal_index] for removal_index, grad in enumerate(grads) if grad is None]
        for node in removed_nodes:
            node.is_active = False
        for node in active_nodes:
            node.check_still_active()
        for node in removal_candidates:
            node.training_variable.assign(1.0)
        self.output_nodes = [node for node in self.output_nodes if node.is_active]
        for node in self.nodes:
            node.was_revived = False
        if 0 == len(self.output_nodes):
            nodes_to_revive = []
            self.output_nodes = [node for node in self.nodes if node.output_field.field_index == self.fields[-1].field_index]
            nodes_to_revive.extend(self.output_nodes)
            for node in self.output_nodes:
                node.is_active = True
                node.was_revived = True
            reviving_nodes = [downstream_node for output_node in self.output_nodes for downstream_node in output_node.accumulate_revival()]
            nodes_to_revive.extend(reviving_nodes)
            while 0 < len(reviving_nodes):
                reviving_nodes = [downstream_node for reviving_node in reviving_nodes for downstream_node in
                                  reviving_node.accumulate_revival()]
                nodes_to_revive.extend(reviving_nodes)
            for node in nodes_to_revive:
                node.is_active = True
                node.was_revived = True
        current_active = [node for node in self.nodes if node.is_active]
        for node in current_active:
            node.check_still_active()
            if not node.is_active:
                raise Exception("Invalid node active!")

        self.op, self.training_variables = self.compile()

    def train(self, training_iter, trainer, training_cycles, pruning_cycles, sample_amount, remove_amount):
        cycle_batch = next(training_iter)
        input_val = cycle_batch["inputs"]
        output_val = cycle_batch["outputs"]
        unscaled_return = self.op(input_val)
        unscaled_return_mu = tf.reduce_mean(unscaled_return)
        unscaled_return_sigma = tf.math.reduce_std(unscaled_return)
        output_return_mu = tf.reduce_mean(output_val)
        output_return_sigma = tf.math.reduce_std(output_val)
        times_array = []
        losses_array = []
        for cycle_index in range(pruning_cycles):
            losses, _, times = self.train_variables(training_cycles, training_iter, unscaled_return_mu, output_return_sigma,
                             unscaled_return_sigma, output_return_mu, trainer)
            times_array.append(np.average(times))
            losses_array.append(losses)
            self.prune_nodes(training_iter, unscaled_return_mu, output_return_sigma, unscaled_return_sigma, output_return_mu, sample_amount, remove_amount)
            if cycle_index < (pruning_cycles / 2):
                for source in (source for source in self.hot_sources if source.node.is_active):
                    source.reward_variable += 1.0
        return np.array(times_array), np.array(losses_array), [source.node is not None and source.node.is_active for source in self.sources]



    def calculate_rewards(self, energy, distance_scalings):
        reward = np.zeros([len(self.sources)])
        if 0 < len(self.hot_sources):
            live_hot_sources = [source for source in self.hot_sources]
            node_reward_scalings = tf.nn.softmax(tf.stack([source.reward_variable for source in live_hot_sources]))


            for source, node_reward_scaling in zip(live_hot_sources, node_reward_scalings):
                reward[source.id] = energy * node_reward_scaling

            live_selectors = [source.id for source in self.sources if not (source.has_graph)]
            selector_cell_scalings = tf.gather(distance_scalings, live_selectors, axis=1)
            tangent_reward_scalings = tf.nn.softmax(tf.gather(selector_cell_scalings, [source.id for source in live_hot_sources]), axis=1)
            node_reward_scalings_ext = tf.expand_dims(node_reward_scalings, 1)
            reward_distance_scalings = tf.multiply(node_reward_scalings_ext, tangent_reward_scalings)
            tangent_rewards = tf.reduce_sum(reward_distance_scalings, axis=0)
            for live_selector_index, tangent_reward in zip(live_selectors, tangent_rewards):
                reward[live_selector_index] = energy * tangent_reward
        return reward

    def refresh_sources(self, indices, cells):
        for cell_index in indices:
            self.sources[cell_index] = NodeSource(cells[cell_index], cell_index)

    def serialize(self):
        return "[" + ", ".join([source.node.serialize() for source in self.hot_sources]) + "]"

    def save_log(self, save_file_name):
        with open(save_file_name, "wb") as save_file:
            pickle.dump(self.log, save_file)
