import numpy as np
import tensorflow as tf
from Field import Field
from GenerateRandomChromosomes import create_random_genome
from Cell import Cell, breed_cells
import random
from CellBatchOperator import CELL_CENTRAL_LAYER_SIZE, CELL_CENTRAL_LAYER_COUNT, operate, update_scalings
from Node import Node
import time
import DatasetProvider
from Graph import Graph


def make_ecosystem(fields_shapes):
    return Ecosystem(fields_shapes)


@tf.function
def tile_to_mutations(gene, generated_family_size):
    return tf.tile(
        tf.reshape(gene, tf.pad(tf.shape(gene), paddings=[[1, 0]], constant_values=1)),
        tf.pad(tf.ones(tf.shape(tf.shape(gene)), dtype=tf.int32), paddings=[[1, 0]],
               constant_values=generated_family_size))

def create_create_mutations(total_mutation_size, total_chomosome_sizes, genome_gene_shapes):
    flat_gene_sizes = [[tf.reduce_prod(gene_size) for gene_size in chromosome_size] for chromosome_size in
                       genome_gene_shapes]

    @tf.function
    def create_mutations(genome, generated_family_size, mutation_rate):
        tiled_genome = [
            [tile_to_mutations(gene, generated_family_size) for gene in chromosome]
            for chromosome in genome]
        total_mutations = tf.random.normal([total_mutation_size], stddev=mutation_rate)
        chromosome_mutations = tf.split(total_mutations, num_or_size_splits=total_chomosome_sizes)
        split_flat_mutations = [tf.split(chromosome_mutation, num_or_size_splits=flat_chromosome_gene_size) for
                                chromosome_mutation, flat_chromosome_gene_size in
                                zip(chromosome_mutations, flat_gene_sizes)]
        gene_mutations = [[tf.reshape(gene_mutation, shape) for gene_mutation, shape in
                           zip(chromosome_and_shape[0], chromosome_and_shape[1])] for chromosome_and_shape in
                          zip(split_flat_mutations, genome_gene_shapes)]
        genome_cells = [
            [tiled_gene + mutation
             for tiled_gene, mutation in zip(chromosome[0], chromosome[1])]
            for chromosome in zip(tiled_genome, gene_mutations)]
        cell_genomes = [[
            [gene_mutation[family_cell_index] for gene_mutation in chromosome]
            for chromosome in genome_cells] for family_cell_index in range(generated_family_size)]
        return cell_genomes

    return create_mutations



class Ecosystem:
    def __init__(self, fields_shapes, initial_cell_groups=1000, generated_family_size=10, total_graph_node_count=1000, mutation_rate=0.01):
        self._fields = tuple(Field(fields_shape, field_index) for field_index, fields_shape in enumerate(fields_shapes))
        self.cells = []
        for field in self._fields:
            field.build_graphs(self._fields)
        self.cell_positions = []
        self.target_graph_size = 512
        last_time = time.time()
        self.total_mutation_size = None
        total_chomosome_sizes = None
        genome_gene_shapes = None
        self.create_mutations = lambda g: None
        self.total_graph_node_count = total_graph_node_count
        self.mutation_rate = mutation_rate
        self.generated_family_size = generated_family_size


        for group_index in range(initial_cell_groups):
            if 0 == group_index % 100:
                print(f"Populating group {group_index}")
            new_cells, new_cell_positions = self.add_random_cells()
            self.cells.extend(new_cells)
            self.cell_positions.extend(new_cell_positions)
        elapsed = time.time() - last_time
        last_time = time.time()

    def add_random_cells(self):
        new_cells = []
        cell_positions = []
        genome = create_random_genome(len(self._fields), hidden_size=CELL_CENTRAL_LAYER_SIZE)
        if self.total_mutation_size is None:
            genome_gene_shapes = [
                [tf.pad(tf.shape(gene), paddings=[[1, 0]], constant_values=self.generated_family_size) for gene in
                 chromosome]
                for chromosome in genome]
            total_chomosome_sizes = [sum(np.prod(gene_size) for gene_size in chromosome) for chromosome in
                                     genome_gene_shapes]
            self.total_mutation_size = sum(total_chomosome_sizes)
            self.create_mutations = create_create_mutations(self.total_mutation_size, total_chomosome_sizes,
                                                       genome_gene_shapes)
        cell_genomes = self.create_mutations(genome, self.generated_family_size, self.mutation_rate)
        group_position = tf.random.uniform([4])
        offsets = tf.random.normal([self.generated_family_size, 4], mean=.01)
        epigene_values = tf.ones([CELL_CENTRAL_LAYER_COUNT, CELL_CENTRAL_LAYER_SIZE], dtype=tf.float32)

        for family_cell_index in range(self.generated_family_size):
            cell_position = group_position + offsets[family_cell_index]
            cell_positions.append(cell_position)
            new_cells.append(
                Cell(self._fields,
                     cell_genomes[family_cell_index],
                     tf.Variable(epigene_values),
                     float(cell_position[0]),
                     float(cell_position[1]),
                     float(cell_position[2]),
                     float(cell_position[3]))
            )
        return new_cells, cell_positions


    def simulate(self, simulation_steps, training_set_file_name, output_file, sample_graph_copy_amount=10, graph_training_cycles=20,
                 total_candidate_used_node_count=500, energy_reward=1000, copy_scale_start=1.1, copy_scale_end=0.8):
        hidden_states = tf.zeros([len(self.cells), CELL_CENTRAL_LAYER_SIZE])
        cell_signals = tf.zeros([len(self.cells), 2, 8])
        cell_positions_tensor = tf.stack(self.cell_positions)
        last_time = time.time()
        SPLITS_FOR_SEGMENTS = 10
        training_set_iter = DatasetProvider.provide_dataset(training_set_file_name, batch_size=256)
        # trainer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
        candidate_graphs = []
        trainer = tf.keras.optimizers.RMSprop()
        distance_scalings = None
        sample_graphs = []


        for step_index in range(simulation_steps):
            print(f"Step {step_index + 1}")
            if 0 == step_index % 10:
                elapsed = time.time() - last_time
                last_time = time.time()
            mating_list = []
            transfer_list = []
            hidden_states, cell_positions_tensor, cell_signals, distance_scalings = operate(self.cells, self._fields, hidden_states,
                                                                         cell_signals, mating_list, transfer_list,
                                                                         cell_positions_tensor, distance_scalings)


            cells_with_graphs = [cell for cell in self.cells if cell.has_graph]
            if 50 < len(cells_with_graphs):
                if 0 == len(candidate_graphs):
                    starting_graph = Graph(self.cells, self._fields, 5, 2000, distance_scalings)
                    candidate_graphs.append(starting_graph)
                    _ = starting_graph.train(training_set_iter, trainer, graph_training_cycles)
                sample_graphs = []
                for graph in candidate_graphs:
                    sample_graphs.extend(graph.copy(sample_graph_copy_amount, copy_scale_start, copy_scale_end, distance_scalings))
                for graph in sample_graphs:
                    graph.train(training_set_iter, trainer, graph_training_cycles)
                sample_graphs.sort(key=lambda sample_graph: sample_graph.evaluation_score)
                candidate_graphs = []
                for scored_graph in sample_graphs:
                    candidate_graphs.append(scored_graph)
                    if sum([len(graph.hot_sources) for graph in candidate_graphs]) > total_candidate_used_node_count:
                        break
                candidate_graph_energy = energy_reward * (1 - tf.nn.softmax([graph.evaluation_score for graph in candidate_graphs]))
                for rewarded_graph, energy in zip(candidate_graphs, candidate_graph_energy):
                    rewarded_graph.allocate_energy(energy, distance_scalings)
            self.manage_energy_transfer(transfer_list, cell_positions_tensor, hidden_states)
            cell_positions_tensor, new_indices = self.mate_cells(mating_list, cell_positions_tensor)
            dead_cell_indexes = [cell_index for cell_index, cell in enumerate(self.cells) if cell.is_dead]
            if self.generated_family_size < len(dead_cell_indexes):
                updating_position_tensor = np.array(cell_positions_tensor)
                while self.generated_family_size < len(dead_cell_indexes):
                    new_cells, new_cell_positions = self.add_random_cells()
                    for new_cell, new_cell_position in zip(new_cells, new_cell_positions):
                        refreshing_index = dead_cell_indexes.pop()
                        self.cells[refreshing_index] = new_cell
                        updating_position_tensor[refreshing_index][0] = new_cell_position[0]
                        updating_position_tensor[refreshing_index][1] = new_cell_position[1]
                        updating_position_tensor[refreshing_index][2] = new_cell_position[2]
                        updating_position_tensor[refreshing_index][3] = new_cell_position[3]
                        new_indices.append(refreshing_index)
                cell_positions_tensor = tf.constant(updating_position_tensor)
            scaling_gather_indices = [index for index in range(len(self.cells))]
            updating_index = len(self.cells)
            for scaling_index in new_indices:
                scaling_gather_indices[scaling_index] = updating_index
                updating_index += 1
            if 0 < len(new_indices):
                distance_scalings = update_scalings(
                    tf.stack([tf.stack([self.cells[refreshed_index].w, self.cells[refreshed_index].x, self.cells[refreshed_index].y, self.cells[refreshed_index].z]) for refreshed_index in new_indices]),
                    cell_positions_tensor,
                    distance_scalings,
                    scaling_gather_indices)
            for updating_graph in candidate_graphs:
                updating_graph.refresh_sources(new_indices, self.cells)
        with open(output_file) as graph_output:
            graph_output.write(sample_graphs[0].serialize())



    def manage_energy_transfer(self, transfer_list, cell_positions_tensor, hidden_states):
        if 0 < len(transfer_list):
            mating_targets_exp = tf.expand_dims(tf.stack([transfer_request[1] for transfer_request in transfer_list]), 1)
            cell_positions_tensor_exp = tf.expand_dims(cell_positions_tensor, 0)
            squared_diffs = tf.square(cell_positions_tensor_exp - mating_targets_exp)
            distances = tf.sqrt(tf.reduce_sum(squared_diffs, axis=-1))

            for transfer_cell_target_and_key, target_cell_index in zip(transfer_list, tf.argmin(distances, axis=0)):
                transfer_cell, target, key = transfer_cell_target_and_key
                target_cell = self.cells[target_cell_index]
                transfer_amount = target_cell.accept_energy(tf.concat([key, hidden_states[target_cell_index]], axis=0))
                transfer_cell.conclude_transmit(transfer_amount)


    def mate_cells(self, mating_list, cell_positions_tensor):
        new_indices = []
        if 0 == len(mating_list):
            return cell_positions_tensor, new_indices
        else:
            mating_targets_exp = tf.expand_dims(tf.stack([mating_request[1] for mating_request in mating_list]), 1)
            cell_positions_tensor_exp = tf.expand_dims(cell_positions_tensor, 0)
            squared_diffs = tf.square(cell_positions_tensor_exp - mating_targets_exp)
            distances = tf.sqrt(tf.reduce_sum(squared_diffs, axis=-1))
            replacement_indexes = [cell_index for cell_index, cell in enumerate(self.cells) if cell.is_dead]
            updating_position_tensor = np.array(cell_positions_tensor)

            for mating_cell_and_target, target_cell_index, replacement_index in zip(mating_list, tf.argmin(distances, axis=0), replacement_indexes):
                if replacement_index is None or mating_cell_and_target is None:
                    break
                target_cell = self.cells[target_cell_index]
                new_genome = breed_cells(mating_cell_and_target[0], target_cell, self.mutation_rate)
                epigene_values = tf.ones([CELL_CENTRAL_LAYER_COUNT, CELL_CENTRAL_LAYER_SIZE], dtype=tf.float32)
                new_cell = Cell(self._fields,
                         new_genome,
                         tf.Variable(epigene_values),
                         mating_cell_and_target[0].w,
                         mating_cell_and_target[0].x,
                         mating_cell_and_target[0].y,
                         mating_cell_and_target[0].z)
                self.cells[replacement_index] = new_cell
                updating_position_tensor[replacement_index][0] = mating_cell_and_target[0].w
                updating_position_tensor[replacement_index][1] = mating_cell_and_target[0].x
                updating_position_tensor[replacement_index][2] = mating_cell_and_target[0].y
                updating_position_tensor[replacement_index][3] = mating_cell_and_target[0].z
                new_indices.append(replacement_index)
            return tf.constant(updating_position_tensor), new_indices






    def profile_field_ops(self):
        return {"field_shift_times": sum(field.field_shift_times for field in self._fields),
                "projection_times": sum(field.projection_times for field in self._fields),
                "softmax_times": sum(field.softmax_times for field in self._fields),
                "einsum_times": sum(field.einsum_times for field in self._fields),
                "conv_times": sum(field.conv_times for field in self._fields),
                "concat_times": sum(field.concat_times for field in self._fields),
                "multiply_times": sum(field.multiply_times for field in self._fields),
                "add_times": sum(field.add_times for field in self._fields),
                "bell_times": sum(field.bell_times for field in self._fields)}


    def test_scaling(self):
        pass
