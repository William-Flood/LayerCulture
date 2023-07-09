import numpy as np
import tensorflow as tf
from Field import Field
from GenerateRandomChromosomes import create_random_genome
from Cell import Cell
import random
from CellBatchOperator import CELL_CENTRAL_LAYER_SIZE, CELL_CENTRAL_LAYER_COUNT, operate
from Node import Node
import time


def make_ecosystem(fields_shapes):
    return Ecosystem(fields_shapes)


class Ecosystem:
    def __init__(self, fields_shapes, initial_cell_groups=1000, generated_family_size=10):
        self._fields = tuple(Field(fields_shape, field_index) for field_index, fields_shape in enumerate(fields_shapes))
        self.cells = []
        for field in self._fields:
            field.build_graphs(self._fields)
        self.cell_positions = []
        self.target_graph_size = 512
        last_time = time.time()
        total_mutation_size = None
        total_chomosome_sizes = None
        genome_gene_shapes = None
        create_mutations = lambda g: None

        @tf.function
        def tile_to_mutations(gene):
            return tf.tile(
                tf.reshape(gene, tf.pad(tf.shape(gene), paddings=[[1, 0]], constant_values=1)),
                tf.pad(tf.ones(tf.shape(tf.shape(gene)), dtype=tf.int32), paddings=[[1, 0]], constant_values=generated_family_size))

        def create_create_mutations(total_mutation_size, total_chomosome_sizes, genome_gene_shapes):
            flat_gene_sizes = [[tf.reduce_prod(gene_size) for gene_size in chromosome_size] for chromosome_size in genome_gene_shapes]

            @tf.function
            def create_mutations(genome):
                tiled_genome = [
                    [tile_to_mutations(gene) for gene in chromosome]
                                 for chromosome in genome]
                total_mutations = tf.random.normal([total_mutation_size])
                chromosome_mutations = tf.split(total_mutations, num_or_size_splits=total_chomosome_sizes)
                split_flat_mutations = [tf.split(chromosome_mutation, num_or_size_splits=flat_chromosome_gene_size) for chromosome_mutation, flat_chromosome_gene_size in zip(chromosome_mutations, flat_gene_sizes)]
                gene_mutations = [[tf.reshape(gene_mutation, shape) for gene_mutation, shape in zip(chromosome_and_shape[0], chromosome_and_shape[1])] for chromosome_and_shape in zip(split_flat_mutations, genome_gene_shapes)]
                genome_cells = [
                    [tiled_gene + mutation
                              for tiled_gene, mutation in zip(chromosome[0], chromosome[1])]
                    for chromosome in zip(tiled_genome, gene_mutations)]
                cell_genomes = [[
                                    [gene_mutation[family_cell_index] for gene_mutation in chromosome]
                                for chromosome in genome_cells] for family_cell_index in range(generated_family_size)]
                return cell_genomes
            return create_mutations


        for group_index in range(initial_cell_groups):
            genome = create_random_genome(len(self._fields), hidden_size=CELL_CENTRAL_LAYER_SIZE)
            if total_mutation_size is None:
                genome_gene_shapes = [[tf.pad(tf.shape(gene), paddings=[[1, 0]], constant_values=generated_family_size) for gene in chromosome]
                             for chromosome in genome]
                total_chomosome_sizes = [sum(np.prod(gene_size) for gene_size in chromosome) for chromosome in genome_gene_shapes]
                total_mutation_size = sum(total_chomosome_sizes)
                create_mutations = create_create_mutations(total_mutation_size, total_chomosome_sizes, genome_gene_shapes)
            cell_genomes = create_mutations(genome)
            group_position = tf.random.uniform([4])
            offsets = tf.random.normal([generated_family_size, 4], mean=.01)
            if 0 == group_index % 100:
                print(f"Populating group {group_index}")
            epigene_values = tf.ones([CELL_CENTRAL_LAYER_COUNT, CELL_CENTRAL_LAYER_SIZE], dtype=tf.float32)

            for family_cell_index in range(generated_family_size):
                cell_position = group_position + offsets[family_cell_index]
                self.cell_positions.append(cell_position)
                self.cells.append(
                    Cell(self._fields,
                         cell_genomes[family_cell_index],
                         # genome,
                         tf.Variable(epigene_values),
                         float(cell_position[0]),
                         float(cell_position[1]),
                         float(cell_position[2]),
                         float(cell_position[3]))
                )
        elapsed = time.time() - last_time
        last_time = time.time()

    def simulate(self, simulation_steps):
        hidden_states = tf.zeros([len(self.cells), CELL_CENTRAL_LAYER_SIZE])
        cell_signals = tf.zeros([len(self.cells), 2, 8])
        cell_positions_tensor = tf.stack(self.cell_positions)
        golden_graph = tuple(tuple() for cell in self.cells)
        platinum_graph = tuple(tuple() for cell in self.cells)
        last_time = time.time()
        SPLITS_FOR_SEGMENTS = 10
        # SPLITS_FOR_SEGMENTS = 2
        segment_accumulator_single = tf.constant([[
            0,
            SPLITS_FOR_SEGMENTS,
            SPLITS_FOR_SEGMENTS ** 2,
            SPLITS_FOR_SEGMENTS ** 3,
        ]], dtype=tf.int32)

        for step_index in range(simulation_steps):
            print(f"Step {step_index + 1}")
            if 0 == step_index % 10:
                elapsed = time.time() - last_time
                last_time = time.time()
            hidden_states, cell_positions_tensor, cell_signals = operate(self.cells, self._fields, hidden_states, cell_signals, [], [], cell_positions_tensor)
            sectors = tuple([] for sector_index in range (SPLITS_FOR_SEGMENTS ** 4))
            cell_sectors = tf.cast(cell_positions_tensor * SPLITS_FOR_SEGMENTS, dtype=tf.int32)
            segment_accumulator = tf.tile(segment_accumulator_single, [len(cell_sectors), 1])
            cell_sector_index = tf.math.mod(tf.reduce_sum(tf.multiply(cell_sectors, segment_accumulator), axis=1), SPLITS_FOR_SEGMENTS ** 4)
            for cell, sector_index in zip(self.cells, cell_sector_index):
                sectors[sector_index].append(cell)

            sector_nodes = []
            for sector_cells in [sector_cells for sector_cells in  sectors if 0 < len(sector_cells)]:
                sector_nodes.append(self.construct_sector_graph(sector_cells))

            full_graph = self.construct_full_graph(tuple(node_and_index for node_and_index in enumerate(sector_nodes) if 0 < len(node_and_index[1])), SPLITS_FOR_SEGMENTS)
            test_value = self.test_graph(full_graph)
            tf.debugging.check_numerics(test_value, "Test failed")

    def profile_field_ops(self):
        return {"field_shift_times": sum(field.field_shift_times for field in self._fields),
            "projection_times":sum(field.projection_times for field in self._fields),
            "softmax_times":sum(field.softmax_times for field in self._fields),
            "einsum_times":sum(field.einsum_times for field in self._fields),
            "conv_times":sum(field.conv_times for field in self._fields),
            "concat_times":sum(field.concat_times for field in self._fields),
            "multiply_times":sum(field.multiply_times for field in self._fields),
            "add_times":sum(field.add_times for field in self._fields),
            "bell_times":sum(field.bell_times for field in self._fields)}

    def construct_sector_graph(self, sector_cells):
        # cell_orderings = [(index, cell.z, cell.is_locked) for index, cell in enumerate(sector_cells) if (cell.has_graph and cell.is_locked)]
        cell_orderings = [(index, cell.z, cell.is_locked) for index, cell in enumerate(sector_cells) if
                          cell.has_graph]
        locked_length = len(tuple(ordering_metadata for ordering_metadata in cell_orderings if ordering_metadata[2]))
        cell_orderings.sort(key=lambda c: c[1])
        field_nodes = tuple([] for _ in self._fields)
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
                field=self._fields[field.field_index],
                op_providers=f"Sector entry {field.field_index}"
            )

        # start_nodes = []
        for field_node_list, field in zip(field_nodes, self._fields):
            start_node = make_start_node(field)
            field_node_list.append(start_node)
            total_nodes.append(start_node)
            # start_nodes.append(start_node)

        for node_index, cell_index in enumerate(cell_index for cell_index, _, _ in cell_orderings):
            cell = sector_cells[cell_index]
            # Cells contain a list of Tensorflow ops, which act on aggregated values emmited to the assembly of fields in the
            # environment.  The last op emits a new value into the environment.  cell.generate_ops is a function that returns
            # a function to loop through those ops.
            cell_op_provider = cell.generate_ops
            cell_node = Node(
                node_index + len(self._fields),
                field_nodes,
                cell_op_provider,
                cell.w , cell.x, cell.y, cell.z,
                cell.visited,
                self._fields[cell.field_index],
                cell
            )
            field_nodes[cell.field_index].append(cell_node)
            total_nodes.append(cell_node)
        # Filter out nodes that don't contribute to the final result of the graph execution
        # Start by setting the node usage to true or false, based on whether it outputs to the global output field, and
        # returning the nodes with edges that point to the nodes that are marked as used
        usage_updates = [marked_node for this_field_node in field_nodes for node in this_field_node for marked_node in node.reset_usage(len(self._fields) - 1)]
        while 0 < len(usage_updates):
            # For each node that had been marked as used in the last usage determination pass, mark the nodes
            # connected to the edges pointing to that node as used, and return the nodes with edges that point to the
            # nodes that are marked as used
            usage_updates = [marked_node for this_field_node in field_nodes for node in this_field_node for marked_node in node.mark_used_edges()]
        # for node in start_nodes:
        #     node.reset_usage(node.output_field.field_index)
        used_nodes = [node for node in total_nodes if node.is_used]
        # start_node_missed = any(start_node not in used_nodes for start_node in start_nodes)
        # for new_index, node in enumerate(used_nodes):
        #     node.index = new_index
        return used_nodes, sector_inputs

    def construct_full_graph(self, sectors, splits_for_segments):
        field_nodes = tuple([] for _ in self._fields)
        this_z_field_nodes = tuple([] for _ in self._fields)
        total_nodes = []
        op_gen_times = []
        for field in self._fields:
            start_node = Node(
                node_index=0,
                candidate_edges=field_nodes,
                node_op_provider=lambda test_mode, fields, fields_emission_positions_lists: lambda node_op_states: None,
                w=0, x=0, y=0, z=0,
                field_visited_check=lambda field_index: field.field_index == field_index,
                field=field,
                op_providers=f"Graph entry"
            )
            field_nodes[field.field_index].append(start_node)
            total_nodes.append(start_node)

        def make_sector_op(sector_nodes, sector_input, total_sector_node_positions):
            '''
            Creates a consolidated function call to load the sector with input values and run through the operations of its nodes
            :param sector_nodes: The nodes within the current sector
            :param sector_input: An input object bound to the starting nodes of the sector
            :param total_sector_node_positions: The location of the nodes in the sector
            :return: A function to call when evaluating the sector's emitted value
            '''
            sector_node_ops = [sector_node.make_node_eval(
                test_mode=True,
                fields=self._fields,
                fields_emission_positions_lists=total_sector_node_positions
            ) for sector_node_index, sector_node in enumerate(sector_nodes)]

            # After the sector node aggregates the environment field values, this function is called
            # to broadcast those values for its internal graph
            # @tf.function
            def emit_sector_value(emisssion_values):
                for field_index, value in emisssion_values.items():
                    sector_input[field_index] = value
                for node_op in sector_node_ops:
                    node_op()
                return sector_nodes[-1].emitted_value

            return lambda: emit_sector_value

        def make_sector_visited(visiting_sector):
            return lambda field_index: any(sector_node.visited_field(field_index) for sector_node in visiting_sector)

        for node_index, sector_and_index in enumerate(sectors):
            sector_index, sector = sector_and_index
            sector_nodes, sector_inputs = sector
            total_sector_node_positions = tf.constant([[node.w, node.x, node.y, node.z] for node in sector_nodes])
            op_gen_time_start = time.time()
            output_field = sector_nodes[-1].output_field

            op_gen_times.append(time.time() - op_gen_time_start)
            sector_w = sector_index % splits_for_segments
            sector_x = (sector_index - sector_w) % (splits_for_segments ** 2)
            sector_y = (sector_index - sector_x * splits_for_segments - sector_w) % (splits_for_segments ** 3)
            sector_z = (sector_index - sector_y * (splits_for_segments ** 2) - sector_x * splits_for_segments - sector_w) % (splits_for_segments ** 3)
            sector_visited = make_sector_visited(sector_nodes)
            sector_node = Node(
                node_index,
                field_nodes,
                make_sector_op(sector_nodes, sector_inputs, total_sector_node_positions),
                sector_w,
                sector_x,
                sector_y,
                sector_z,
                sector_visited,
                output_field,
                sector_nodes
            )
            this_z_field_nodes[output_field.field_index].append(sector_node)
            total_nodes.append(sector_node)
            # Ensures that sectors are only able to form connections with sectors with a lower z-index than them
            if 0 == ((sector_index + 1) % splits_for_segments ** 3):
                for field_node, this_z_field_node in zip(field_nodes, this_z_field_nodes):
                    field_node.extend(this_z_field_node)

        # Filter out nodes that don't contribute to the final result of the graph execution
        # Start by setting the node usage to true or false, based on whether it outputs to the global output field, and
        # returning the nodes with edges that point to the nodes that are marked as used
        usage_updates = [marked_node for node in total_nodes for marked_node in node.reset_usage(len(self._fields) - 1)]
        while 0 < len(usage_updates):
            # For each node that had been marked as used in the last usage determination pass, mark the nodes
            # connected to the edges pointing to that node as used, and return the nodes with edges that point to the
            # nodes that are marked as used
            usage_updates = [marked_node for node in total_nodes for marked_node in node.mark_used_edges()]
        # Graph execution requires that the input fields of the full graph exist
        for field in self._fields:
            total_nodes[field.field_index].reset_usage(field.field_index)
        used_nodes = [node for node in total_nodes if node.is_used]
        for new_index, node in enumerate(used_nodes):
            node.index = new_index
        return used_nodes

    def test_graph(self, full_graph):
        total_sector_node_positions = tf.constant([[node.w, node.x, node.y, node.z] for node in full_graph])
        batch_size = 16
        # sample_input = tf.zeros([batch_size] + self._fields[0].shape)
        node_field_indexes = []
        field_nodes = [0 for field in self._fields]
        for node in full_graph:
            node_field_indexes.append(field_nodes[node.output_field.field_index])
            field_nodes[node.output_field.field_index] = field_nodes[node.output_field.field_index] + 1
        for field in self._fields:
            if 0 == field_nodes[field.field_index]:
                field_nodes[field.field_index] = 1
        graph_build_start = time.time()
        node_ops_and_vars = [node.make_node_eval(
            test_mode=True,
            fields=self._fields,
            fields_emission_positions_lists=total_sector_node_positions
        ) for node in full_graph[len(self._fields):]]
        graph_build_time = time.time() - graph_build_start
        field_emission_values = [tf.zeros([emitting_count, batch_size] + field.shape) for emitting_count, field in zip(field_nodes, self._fields)]
        # pad_to = [[0, field_nodes[0] - 1], [0, 0]] + [[0, 0]] * len(self._fields[0].shape)
        # field_emission_values[0] = tf.pad(tf.reshape(sample_input, [1] + [batch_size] + self._fields[0].shape), pad_to)
        # full_graph[0].emitted_value = sample_input
        for field in self._fields:
            full_graph[field.field_index].emitted_value = tf.zeros([16] + field.shape)

        for field in self._fields:
            field.reset_op_times()

        def run_graph():
            for op in node_ops_and_vars:
                op()
            return full_graph[-1].emitted_value

        graph_start = time.time()
        return_val = run_graph()
        elapsed = time.time() - graph_start
        return return_val

