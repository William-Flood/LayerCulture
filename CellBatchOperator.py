import tensorflow as tf
import numpy as np
from CellConstants import ChromosomeSet, ActionSet, CELL_CENTRAL_LAYER_COUNT, CELL_CENTRAL_LAYER_SIZE, GENERATION_KEY_SIZE, ENVIRONMENT_DIMENSIONS
from typing import Tuple, List
from Field import Field
from Cell import Cell
import time


def generate_cell_context(cell_signal_field_values, cells):
    w_1, b_1, w_2, b_2 = \
        batch_genes(cells, ChromosomeSet.RECEPTORS, 4)
    receptor_hidden = tf.nn.relu(tf.einsum("cds,cdso->co", cell_signal_field_values, w_1) + b_1)
    receptor_outputs = tf.einsum("cv,cvo->co", receptor_hidden, w_2) + b_2
    return receptor_outputs

def batch_genes(cells, chromosome : ChromosomeSet, chromosome_size):
    cell_genes = tuple(cell.provide_chromosome(chromosome) for cell in cells)
    for gene_index in range(chromosome_size):
        yield tf.stack(tuple(gene[gene_index] for gene in cell_genes))


def update_hidden(hidden_states, cell_signal_field_values, cells, cell_epigenetics):
    cell_internal_context_matrix = tf.stack([cell.provide_signal_input() for cell in cells])
    state_0 = tf.concat((hidden_states, cell_signal_field_values, cell_internal_context_matrix), axis=1)
    w_1, b_1, w_2, b_2, w_3, b_3, w_4, b_4, w_5, b_5, w_6, b_6, w_7, b_7, w_8, b_8 = \
        batch_genes(cells, ChromosomeSet.MAIN, 16)
    state_1 = tf.nn.relu(tf.multiply(tf.einsum('cik,ci->ck', w_1, state_0) + b_1, cell_epigenetics[:, 0, :]))
    state_2 = tf.math.atan(
        tf.nn.relu(tf.multiply(tf.einsum('cik,ci->ck', w_2, state_1) + b_2, cell_epigenetics[:, 1, :])))
    state_3 = tf.nn.relu(tf.multiply(tf.einsum('cik,ci->ck', w_3, state_2) + b_3, cell_epigenetics[:, 2, :]))
    state_4 = tf.math.atan(
        tf.nn.relu(tf.multiply(tf.einsum('cik,ci->ck', w_4, state_3) + b_4, cell_epigenetics[:, 3, :])))
    state_5 = tf.nn.relu(tf.multiply(tf.einsum('cik,ci->ck', w_5, state_4) + b_5, cell_epigenetics[:, 4, :]))
    state_6 = tf.math.atan(
        tf.nn.relu(tf.multiply(tf.einsum('cik,ci->ck', w_6, state_5) + b_6, cell_epigenetics[:, 5, :])))
    state_7 = tf.nn.relu(tf.multiply(tf.einsum('cik,ci->ck', w_7, state_6) + b_7, cell_epigenetics[:, 6, :]))
    state_8 = tf.math.atan(tf.multiply(tf.einsum('cik,ci->ck', w_8, state_7) + b_8, cell_epigenetics[:, 7, :]))
    return state_8

# @tf.function
def batch_precalc_genes(precalc_genes, action_selection):
    w1_array = []
    b1_array = []
    w2_array = []
    b2_array = []
    for cell_precalc_genes in precalc_genes:
        w1_array.append(cell_precalc_genes[0])
        b1_array.append(cell_precalc_genes[1])
        w2_array.append(cell_precalc_genes[2])
        b2_array.append(cell_precalc_genes[3])

    return tf.gather(tf.stack(w1_array), action_selection, axis=1, batch_dims=1), \
       tf.gather(tf.stack(b1_array), action_selection, axis=1, batch_dims=1), \
       tf.gather(tf.stack(w2_array), action_selection, axis=1, batch_dims=1), \
       tf.gather(tf.stack(b2_array), action_selection, axis=1, batch_dims=1)

def select_action(hidden_states, cells):
    cell_action_filters = tf.stack([cell.action_mask for cell in cells])
    w1, b1, w2, b2, w3, b3 = batch_genes(cells, ChromosomeSet.SELECT, 6)
    selection_state_1 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1))
    selection_state_2 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w2, selection_state_1) + b2))
    selection_heat = tf.nn.sigmoid(tf.nn.relu(tf.einsum('cik,ci->ck', w3, selection_state_2) + b3)) + 0.01
    masked_selection_heats = tf.multiply(selection_heat, cell_action_filters)
    # Selects the action with the highest selection favorability calculation within the permitted actions for the cell.
    action_selections = tf.argmax(masked_selection_heats, axis=1, output_type=tf.int32)

    action_w1, action_b1, action_w2, action_b2 = batch_precalc_genes(
        (cell.provide_chromosome(ChromosomeSet.ACTION_PRECALCULATION_SET) for cell in cells),
        action_selections)
    state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', action_w1, hidden_states) + action_b1)
    action_instructions_precalculation = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', action_w2, state_1) + action_b2))
    return action_selections, action_instructions_precalculation


def perform_reset(hidden_state_list, cells):
    if 0 < len(hidden_state_list):
        action_states = tf.stack(hidden_state_list, axis=0)
        w1, b1 = batch_genes(cells, ChromosomeSet.DOUBLE_FIELD_SELECTOR, 2)
        state_1 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w1, action_states) + b1))
        for cell, field_selection in zip(cells, tf.argmax(state_1, axis=1)):
            cell.reset()


def perform_field_shift(hidden_state_list, cells, fields: Tuple[Field]):
    if 0 < len(hidden_state_list):
        operating_field_options = np.stack([field.action_mask for field in fields])[:, ActionSet.FIELD_SHIFT]
        action_states = tf.stack(hidden_state_list, axis=0)
        w1, b1 = batch_genes(cells, ChromosomeSet.DOUBLE_FIELD_SELECTOR, 2)
        state_1 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w1, action_states) + b1))
        normalized_field_shift_values = tf.math.sigmoid(state_1) + .01
        operating_field_selection_values, target_field_selection_values = tf.split(normalized_field_shift_values, num_or_size_splits=2, axis=1)
        operating_field_heat_map = tf.multiply(operating_field_selection_values, operating_field_options)
        for cell, operating_field_selection, target_field_selection in zip(
                cells,
                tf.argmax(operating_field_heat_map, axis=1),
                target_field_selection_values
        ):
            field_with = tf.argmax(
                tf.multiply(
                        target_field_selection,
                        fields[int(operating_field_selection)].field_shift_allowances
            ))
            cell.add_field_shift(int(operating_field_selection), int(field_with))


def perform_projection(hidden_state_list, cells, fields: Tuple[Field]):
    if 0 < len(hidden_state_list):
        operating_field_options = np.stack([field.action_mask for field in fields])[:, ActionSet.PROJECTION]
        action_states = tf.stack(hidden_state_list, axis=0)
        w1, b1 = batch_genes(cells, ChromosomeSet.DOUBLE_FIELD_SELECTOR_AND_KEY, 2)
        state_1 = tf.einsum('cik,ci->ck', w1, action_states) + b1
        projection_values, keys = tf.split(state_1, [len(fields) * 2, 16], axis=1)
        normalized_projection_values = tf.math.sigmoid(projection_values) + .01
        operating_field_selection_values, target_field_selection_values = tf.split(normalized_projection_values, num_or_size_splits=2, axis=1)
        operating_field_heat_map = tf.multiply(operating_field_selection_values, operating_field_options)
        for cell, operating_field_selection, target_field_selection, cell_key in zip(
                    cells,
                    tf.argmax(operating_field_heat_map, axis=1),
                    target_field_selection_values,
                    keys
        ):
            field_to = tf.argmax(
                tf.multiply(
                        target_field_selection,
                        fields[int(operating_field_selection)].projection_allowances
                )
            )
            cell.add_projection(int(operating_field_selection), int(field_to), cell_key)


def perform_softmax(hidden_state_list, cells):
    if 0 < len(hidden_state_list):
        action_states = tf.stack(hidden_state_list, axis=0)
        w1, b1 = batch_genes(cells, ChromosomeSet.SINGLE_FIELD_SELECTOR, 2)
        state_1 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w1, action_states) + b1))
        operating_field_heat_map = tf.math.sigmoid(state_1) + .01
        for cell, operating_field_selection in zip(
                cells,
                tf.argmax(operating_field_heat_map, axis=1)
        ):
            cell.add_softmax(int(operating_field_selection))

def perform_bell(hidden_state_list, cells):
    if 0 < len(hidden_state_list):
        action_states = tf.stack(hidden_state_list, axis=0)
        w1, b1 = batch_genes(cells, ChromosomeSet.SINGLE_FIELD_SELECTOR, 2)
        state_1 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w1, action_states) + b1))
        operating_field_heat_map = tf.math.sigmoid(state_1) + .01
        for cell, operating_field_selection in zip(
                cells,
                tf.argmax(operating_field_heat_map, axis=1)
        ):
            cell.add_bell(int(operating_field_selection))

def perform_einsum(hidden_state_list, cells, fields: Tuple[Field]):
    if 0 < len(hidden_state_list):
        operating_field_options = np.stack([field.action_mask for field in fields])[:, ActionSet.EINSUM]
        action_states = tf.stack(hidden_state_list, axis=0)
        w1, b1 = batch_genes(cells, ChromosomeSet.TRIPLE_FILED_SELECTOR, 2)
        state_1 = tf.einsum('cik,ci->ck', w1, action_states) + b1
        normalized_state_1 = tf.math.sigmoid(state_1) + .01
        operating_field_selection_source, field_with_selection_source, field_to_selection_source = tf.split(normalized_state_1, 3, axis=1)
        operating_field_heat_map = tf.multiply(field_with_selection_source, operating_field_options)
        for cell, operating_field, cell_normalized_einsum_with_values, cell_normalized_einsum_to_values in zip(
                cells,
                tf.argmax(operating_field_heat_map, axis=1),
                field_with_selection_source,
                field_to_selection_source
        ):
            field_selection_with = tf.argmax(
                                 tf.multiply(
                                     cell_normalized_einsum_with_values,
                                     fields[int(operating_field)].einsum_allowances
                                 )
                             )
            cell_einsum_allowances_with = fields[int(operating_field)].einsum_allowances_with[int(field_selection_with)]
            field_shift_heat_map_to = tf.multiply(cell_normalized_einsum_to_values, cell_einsum_allowances_with)
            field_selection_to = tf.argmax(field_shift_heat_map_to)
            cell.add_einsum(int(operating_field), int(field_selection_with), int(field_selection_to))


def perform_conv(hidden_state_action_list, cells, fields: Tuple[Field]):
    """
    Updates a deep learning graph contained within a list of cells.

    This function takes a list of hidden states, cells, and fields as input, and calculates the preference
    of each organism to generate an output to each field using a series of einsum operations and
    activations. It then adds a convolution layer to each organism, which will output to the selected
    field using weights and biases generated pseudorandomly based on a kernel generation seed.

    Args:
    hidden_state_list (List[tf.Tensor]): A list of hidden states for each cell.
    cells (List[Cell]): A list of cells (organisms) in the genetic algorithm.
    fields (List[Field]): A list of fields used to store tensors; these serve as the inputs and outputs
    for operations in a deep learning graph

    Returns:
    None. The function updates the cells by adding a convolution layer to each of them.
    """
    if 0 < len(hidden_state_action_list):
        # Gets the list of fields which offer the option of a convolution operation
        operating_field_options = np.stack([field.action_mask for field in fields])[:, ActionSet.CONV]
        # Consolidate the hidden states provided for each cell into a single tensor to be operated on in a single operation
        action_states = tf.stack(hidden_state_action_list, axis=0)
        # Get the parameters used in the instructions to add convolution operations to organisms
        w1, b1 = batch_genes(cells, ChromosomeSet.DOUBLE_FIELD_SELECTOR_AND_KEY, 2)
        # Finalize the instructions by computing the einsum between the cells' precalculated action state and a weight tensor, and adding a bias tensor
        state_1 = tf.einsum('cik,ci->ck', w1, action_states) + b1
        # Break out the components of the chromosome evaluation for their separate roles in building the cell graph
        field_with_selection_source, kernel_generation_seed = tf.split(state_1, [len(fields) * 2, 2 * GENERATION_KEY_SIZE], axis=1)
        # Ensure that all values in the heat map used to select the target field of the convolution operation are positive
        # This ensures that the selected field is contained in the selection masks
        normalized_field_to_selection_source = tf.math.sigmoid(field_with_selection_source) + .01
        # Split the selection sources into two sets, one for the operating field and one for the target field
        operating_field_selection_values, target_field_selection_values = tf.split(normalized_field_to_selection_source, num_or_size_splits=2, axis=1)
        # Filters operating_field_selection_values by the list of options to compute the cell's preference for each
        # field as the operating field
        operating_field_heat_map = tf.multiply(operating_field_selection_values, operating_field_options)
        # Loop through each cell and its corresponding data
        for cell, operating_field_selection, cell_target_field_selection_values, cell_kernel_generation_seed in zip(cells, tf.argmax(operating_field_heat_map, axis=1), target_field_selection_values, kernel_generation_seed):
            # Find the field that the cell most prefers to set the convolution operation output to
            field_to = tf.argmax(
                tf.multiply(
                        cell_target_field_selection_values,
                        fields[int(operating_field_selection)].projection_allowances
                )
            )
            # Add a convolution layer to the organism, which will output to the field indicated by field_selection and
            # use weights and biases generated pseudorandomly using the value in kernel_generation_seed
            # The cell's field_index property will be updated to the value of field_selection
            cell.add_conv(int(operating_field_selection), int(field_to), cell_kernel_generation_seed)

@tf.function
def update_distances(changed_positions, all_positions):
    changed_positions_ex = tf.expand_dims(changed_positions, axis=0)
    new_positions_ex = tf.expand_dims(all_positions, axis=1)
    squared_diffs = tf.square(changed_positions_ex - new_positions_ex)
    distances = tf.sqrt(tf.reduce_sum(squared_diffs, axis=-1))
    return distances

@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.int32)
    ]
)
def scalings_from_distances(distances, old_distance_scalings, scaling_gather_indices):
    distance_scalings_updates = 1 / tf.square(1 + distances)
    distance_scalings_concat_1 = tf.concat([old_distance_scalings, distance_scalings_updates], axis=1)
    distance_scalings_update_1 = tf.gather(distance_scalings_concat_1, scaling_gather_indices, axis=1)
    distance_scalings_concat_2 = tf.concat([tf.transpose(distance_scalings_update_1, [1, 0]), distance_scalings_updates], axis=1)
    unmasked_updated_scalings = tf.gather(distance_scalings_concat_2, scaling_gather_indices, axis=1)
    updated_scalings = tf.multiply(unmasked_updated_scalings, 1.0 - tf.eye(tf.shape(unmasked_updated_scalings)[0]))
    return updated_scalings

def update_scalings(changed_positions, all_positions, old_distance_scalings, scaling_gather_indices):
    distances = update_distances(changed_positions, all_positions)
    return scalings_from_distances(distances, old_distance_scalings, scaling_gather_indices)

@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, ENVIRONMENT_DIMENSIONS], dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.int32)
    ]
)
def update_positions_and_scalings(state_1, original_positions, move_gather_arg, updating_indexes, old_distance_scalings, scaling_gather_indices):
    static_and_deltas = tf.concat([tf.zeros([1, 4]), state_1], axis=0)
    new_positions = (original_positions + tf.gather(static_and_deltas, move_gather_arg, axis=0)) % 1.0
    updated_distances = update_distances(tf.gather(new_positions, updating_indexes), new_positions)
    new_distance_scalings = scalings_from_distances(updated_distances, old_distance_scalings, scaling_gather_indices)
    return new_positions, new_distance_scalings


def perform_move(hidden_state_action_list, cells, origial_positions, move_gather_arg, updating_indexes, distance_scalings, scaling_gather_indices):
    action_states = tf.stack(hidden_state_action_list, axis=0)
    w1, b1 = batch_genes(cells, ChromosomeSet.DIRECTION_SELECTOR, 2)
    # Finalize the instructions by computing the einsum between the cells' precalculated action state and a weight tensor, and adding a bias tensor
    state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, action_states) + b1)
    for cell, direction in zip(cells, state_1):
        cell.move(direction[0], direction[1], direction[2], direction[3])

    new_positions, new_distance_scalings = update_positions_and_scalings(state_1, origial_positions, tf.constant(move_gather_arg), tf.constant(updating_indexes), distance_scalings, tf.constant(scaling_gather_indices))
    return new_positions, new_distance_scalings


def perform_mate(hidden_state_list, cells, mating_list):
    if 0 < len(hidden_state_list):
        action_states = tf.stack(hidden_state_list, axis=0)
        w1, b1 = batch_genes(cells, ChromosomeSet.DIRECTION_SELECTOR, 2)
        state_1 = tf.einsum('cik,ci->ck', w1, action_states) + b1
        for cell, direction in zip(cells, state_1):
            if cell.mate():
                mating_list.append((cell, direction + tf.constant(np.array([cell.w, cell.x, cell.y, cell.z], dtype=np.float32))))


def perform_add_epigene(hidden_state_list, cells):
    if 0 < len(hidden_state_list):
        action_states = tf.stack(hidden_state_list, axis=0)
        w1, b1 = batch_genes(cells, ChromosomeSet.SELECT_EPIGENE, 2)
        state_2 = tf.einsum('cik,ci->ck', w1, action_states) + b1
        layer_select_heat_map, index_select_heat_map = tf.split(state_2, [CELL_CENTRAL_LAYER_COUNT, CELL_CENTRAL_LAYER_SIZE], axis=1)
        cells_layer_selection = tf.argmax(layer_select_heat_map, axis=1)
        cells_index_selection = tf.argmax(index_select_heat_map, axis=1)
        for cell, layer_selection, index_selection in zip(cells, cells_layer_selection, cells_index_selection):
            cell.add_epigene(int(layer_selection), int(index_selection))


def perform_remove_epigene(hidden_state_list, cells):
    if 0 < len(hidden_state_list):
        action_states = tf.stack(hidden_state_list, axis=0)
        w1, b1 = batch_genes(cells, ChromosomeSet.SELECT_EPIGENE, 2)
        state_1 = tf.einsum('cik,ci->ck', w1, action_states) + b1
        layer_select_heat_map, cells_index_select_heat_map = tf.split(state_1, [CELL_CENTRAL_LAYER_COUNT, CELL_CENTRAL_LAYER_SIZE], axis=1)
        cells_layer_selection = tf.argmax(layer_select_heat_map, axis=1)
        for cell, layer_selection, index_heat_map in zip(cells, cells_layer_selection, cells_index_select_heat_map):
            cell.subtract_epigene(int(layer_selection), index_heat_map)


def perform_transfer_energy(hidden_state_list, cells, transfer_list):
    if 0 < len(hidden_state_list):
        action_states = tf.stack(hidden_state_list, axis=0)
        w1, b1 = batch_genes(cells, ChromosomeSet.DIRECTION_SELECTOR_AND_KEY, 2)
        state_1 = tf.einsum('cik,ci->ck', w1, action_states) + b1
        cells_direction, cells_key = tf.split(state_1, [4, 16], axis=1)
        for cell, direction, key in zip(cells, cells_direction, cells_key):
            transfer_list.append((cell, direction + tf.constant(np.array([cell.w, cell.x, cell.y, cell.z], dtype=np.float32)), key))


def perform_concat(hidden_state_list, cells, fields: Tuple[Field]):
    if 0 < len(hidden_state_list):
        operating_field_options = np.stack([field.action_mask for field in fields])[:, ActionSet.CONCAT]
        action_states = tf.stack(hidden_state_list, axis=0)
        w1, b1 = batch_genes(cells, ChromosomeSet.TRIPLE_FILED_SELECTOR, 2)
        state_1 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w1, action_states) + b1))
        normalized_state_1 = tf.math.sigmoid(state_1) + .01
        operating_field_selection_source, field_with_selection_source, field_to_selection_source = tf.split(normalized_state_1, 3, axis=1)
        operating_field_heat_map = tf.multiply(field_with_selection_source, operating_field_options)
        for cell, operating_field, cell_normalized_einsum_with_values, cell_normalized_einsum_to_values in zip(
                        cells,
                        tf.argmax(operating_field_heat_map, axis=1),
                        field_with_selection_source,
                        field_to_selection_source):
            field_selection_with = tf.argmax(
                                 tf.multiply(
                                     cell_normalized_einsum_with_values,
                                     fields[int(operating_field)].concat_allowances
                                 )
                             )
            cell_concat_allowances_with = fields[int(operating_field)].concat_to_allowances(int(field_selection_with))
            field_shift_heat_map_to = tf.multiply(cell_normalized_einsum_to_values, cell_concat_allowances_with)
            field_selection_to = tf.argmax(field_shift_heat_map_to)

            cell.add_concat(int(operating_field), int(field_selection_with), int(field_selection_to))

def perform_multiply(hidden_state_list, cells, fields):
    """
    Updates a deep learning graph contained within a list of cells.

    This function takes a list of hidden states, cells, and fields as input, and calculates the preference
    of each organism to generate an output to each field using a series of einsum operations and
    activations. It then adds a multiply operation to the deep learning graph managed
    by the cell, multiplying the value of the field indicated by the cell's field_index by
    the value of the field selected within this function for each cell

    Args:
    hidden_state_list (List[tf.Tensor]): A list of hidden states for each cell.
    cells (List[Cell]): A list of cells (organisms) in the genetic algorithm.
    fields (List[Field]): A list of fields used to store tensors; these serve as the inputs and outputs
    for operations in a deep learning graph

    Returns:
    None. The function updates the cells by adding a multiplication layer to each of them.
    """
    if 0 < len(hidden_state_list):
        operating_field_options = np.stack([field.action_mask for field in fields])[:, ActionSet.MULTIPLY]
        # Consolidate the hidden states provided for each cell into a single tensor to be operated on in a single operation
        action_states = tf.stack(hidden_state_list, axis=0)
        # Get the parameters used in the instructions to add convolution operations to organisms
        w1, b1 = batch_genes(cells, ChromosomeSet.DOUBLE_FIELD_SELECTOR, 2)
        # Use the feched parameters to finish calculating the instructions for the operation
        state_1 = tf.einsum('cik,ci->ck', w1, action_states) + b1
        normalized_field_selection_source = tf.math.sigmoid(state_1) + .01
        # Splits the selection sources by role
        operating_field_selection_values, target_field_selection_values = tf.split(normalized_field_selection_source, num_or_size_splits=2, axis=1)
        # Filters operating_field_selection_values by the list of options to compute the cell's preference for each
        # field as the operating field
        operating_field_heat_map = tf.multiply(operating_field_selection_values, operating_field_options)
        # Loop through each cell and its corresponding data
        for cell, operating_field_selection, cell_target_field in zip(cells, tf.argmax(operating_field_heat_map, axis=1), target_field_selection_values):
            # Find the field that the cell most prefers to set the multiply operation output to
            field_with = tf.argmax(
                tf.multiply(
                        cell_target_field,
                        fields[int(operating_field_selection)].elementwise_allowances
                )
            )
            # Add a convolution layer to the organism, which will output to the field indicated by field_selection and
            # use weights and biases generated pseudorandomly using the value in kernel_generation_seed
            # The cell's field_index property will be updated to the value of field_selection
            cell.add_multiply(int(operating_field_selection), int(field_with))

def perform_add(hidden_state_list, cells, fields):
    """
    Updates a deep learning graph contained within a list of cells.

    This function takes a list of hidden states, cells, and fields as input, and calculates the preference
    of each organism to generate an output to each field using a series of einsum operations and
    activations. It then adds a division operation to the deep learning graph managed
    by the cell, dividing the value of the field indicated by the cell's field_index by
    the value of the field selected within this function for each cell

    Args:
    hidden_state_list (List[tf.Tensor]): A list of hidden states for each cell.
    cells (List[Cell]): A list of cells (organisms) in the genetic algorithm.
    fields (List[Field]): A list of fields used to store tensors; these serve as the inputs and outputs
    for operations in a deep learning graph

    Returns:
    None. The function updates the cells by adding a convolution layer to each of them.
    """
    if 0 < len(hidden_state_list):
        operating_field_options = np.stack([field.action_mask for field in fields])[:, ActionSet.ADD]
        # Consolidate the hidden states provided for each cell into a single tensor to be operated on in a single operation
        action_states = tf.stack(hidden_state_list, axis=0)
        # Get the parameters used in the instructions to add convolution operations to organisms
        w1, b1 = \
            batch_genes(cells, ChromosomeSet.DOUBLE_FIELD_SELECTOR, 2)
        # Finalize the instructions by computing the einsum between the cells' precalculated action state and a weight tensor, and adding a bias tensor
        state_3 = tf.einsum('cik,ci->ck', w1, action_states) + b1
        normalized_field_selection_source = tf.math.sigmoid(state_3) + .01
        # Splits the selection sources by role
        operating_field_selection_values, target_field_selection_values = tf.split(normalized_field_selection_source, num_or_size_splits=2, axis=1)
        # Filters operating_field_selection_values by the list of options to compute the cell's preference for each
        # field as the operating field
        operating_field_heat_map = tf.multiply(operating_field_selection_values, operating_field_options)
        # Loop through each cell and its corresponding data
        for cell, operating_field_selection, cell_target_field_selection_values in zip(cells, tf.argmax(operating_field_heat_map, axis=1), target_field_selection_values):
            # Find the field that the cell most prefers to set the add operation output to
            field_with = tf.argmax(
                tf.multiply(
                        cell_target_field_selection_values,
                        fields[int(operating_field_selection)].elementwise_allowances
                )
            )
            # Add a convolution layer to the organism, which will output to the field indicated by field_selection and
            # use weights and biases generated pseudorandomly using the value in kernel_generation_seed
            # The cell's field_index property will be updated to the value of field_selection
            cell.add_add(int(operating_field_selection), int(field_with))

@tf.function
def calculate_masked_distances(distance_scaling, cell_count, cell_positions):
    cell_ones = tf.ones([cell_count, cell_count], dtype=tf.float32)
    zs_extended = tf.expand_dims(cell_positions[:,3], axis=1)
    zs_extended_transpose = tf.transpose(zs_extended,[1, 0])
    front_mask = tf.sign(zs_extended - zs_extended_transpose)
    back_mask = cell_ones - front_mask
    front_back_masked_distances = tf.stack(
        (
            tf.multiply(distance_scaling, front_mask),
            tf.multiply(distance_scaling, back_mask)
        ),
        axis=1
    )
    return front_back_masked_distances

def perform_signal(cells, hidden_states, cell_positions, distance_scalings):
    if 0 < len(hidden_states):
        # distance_calc_start = time.time()
        cell_count = len(cells)
        front_back_masked_distances = calculate_masked_distances(distance_scalings, tf.constant(cell_count), cell_positions)
        # batch_start = time.time()
        # distance_calc_duration = batch_start - distance_calc_start
        w1, b1, w2, b2 = \
            batch_genes(cells, ChromosomeSet.EMIT, 4)
        # signal_calc_start = time.time()
        # batch_duration = signal_calc_start - batch_start
        # Compute state_1 by applying a ReLU activation on an einsum between w1 and hidden_states added to b1
        state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1)
        # Compute state_2 by applying an atan activation on the ReLU activation of an einsum between  w2 and state_1 added to b2
        state_2 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w2, state_1) + b2))
        cell_signal_inputs = tf.einsum("cbd,dk->cbk", front_back_masked_distances, state_2)
        # signal_duration = time.time() - signal_calc_start
        return cell_signal_inputs

def operate(cells: List[Cell], fields : Tuple[Field], cell_hidden_states, cell_signal_values, mating_list, transfer_list, cell_positions, cell_distance_scalings):
    """
        Simulates one time-step of cell operations within the ecosystem.

        Args:
            cells (list[Cell]): A list of cells in the ecosystem.
            fields (Tuple[Field]): A tuple of fields within which cells operate.
            cell_hidden_states (tf.Tensor): Current hidden states of all cells.
            cell_signal_values (tf.Tensor): Current signal values of all cells.
            mating_list (list): A list to capture cells that wish to mate this timestep.
            transfer_list (list): A list to capture cells that wish to transfer energy this timestep.
            cell_positions (tf.Tensor): Current positions of all cells within the ecosystem.
            cell_distance_scalings (tf.Tensor): Distance scalings of cells.

        Returns:
            tuple: Updated values of:
                - cell_hidden_states (tf.Tensor): New hidden states of cells post operation.
                - cell_positions (tf.Tensor): New positions of cells post operation.
                - cell_signal_values (tf.Tensor): New signal values of cells post operation.
                - cell_distance_scalings (tf.Tensor): Updated distance scalings.

        Notes:
            This function simulates the activities of each cell for one timestep based on their internal states,
            field conditions, and interactions with other cells. Each cell can perform a variety of operations such
            as moving, mating, and transferring energy, based on its current state and the action it decides to
            take. The resulting state of each cell, and their new positions in the field, are updated and returned.
    """
    # times.append(time.time())
    context_hints = generate_cell_context(
        cell_signal_values,
        cells
    )
    cell_epigenetics = tf.stack([cell.epigenetics for cell in cells], axis=0)
    new_hidden = update_hidden(cell_hidden_states, context_hints, cells, cell_epigenetics)
    cell_selections, action_states = select_action(new_hidden, cells)
    reset_data = []
    field_shift_data = []
    projection_data = []
    softmax_data = []
    einsum_data = []
    conv_data = []
    move_data = []
    mate_data = []
    add_epigene_data = []
    remove_epigene_data = []
    lock_data = []
    unlock_data = []
    transfer_energy_data = []
    wait_hole = []
    concat_data = []
    multiply_data = []
    add_data = []
    bell_data = []
    data_set = (
        reset_data,
        field_shift_data,
        projection_data,
        softmax_data,
        einsum_data,
        conv_data,
        move_data,
        mate_data,
        add_epigene_data,
        remove_epigene_data,
        lock_data,
        unlock_data,
        transfer_energy_data,
        wait_hole,
        concat_data,
        multiply_data,
        add_data,
        bell_data
    )
    # wait
    move_gather_arg = np.zeros([len(cells)], dtype=np.int32)
    move_counter = 0
    # times.append(time.time())

    move_scaling_gather_indices = [index for index in range(len(cells))]
    move_updating_indexes = []
    for cell, selection, index in zip(cells, cell_selections, range(len(cells))):
        data_set[selection].append((cell, index))
        if(selection == int(ActionSet.MOVE)):
            move_gather_arg[index] = move_counter + 1
            move_scaling_gather_indices[index] = len(cells) + move_counter
            move_counter += 1
            move_updating_indexes.append(index)
    # times.append(time.time())
    perform_reset(
        tf.gather(action_states, np.array(tuple(data[1] for data in reset_data), dtype=np.int32)),
        tuple(data[0] for data in reset_data)
    )
    # times.append(time.time())
    # actions = ["reset"]
    # action_cell_counts = [len(reset_data)]
    perform_field_shift(
        tf.gather(action_states, np.array(tuple(data[1] for data in field_shift_data), dtype=np.int32)),
        tuple(data[0] for data in field_shift_data),
        fields
    )
    # times.append(time.time())
    # actions.append("shift")
    # action_cell_counts.append(len(field_shift_data))
    perform_projection(
        tf.gather(action_states, np.array(tuple(data[1] for data in projection_data), dtype=np.int32)),
        tuple(data[0] for data in projection_data),
        fields
    )
    # times.append(time.time())
    # actions.append("projection")
    # action_cell_counts.append(len(projection_data))
    perform_softmax(
        tf.gather(action_states, np.array(tuple(data[1] for data in softmax_data), dtype=np.int32)),
        tuple(data[0] for data in softmax_data)
    )

    # times.append(time.time())
    # actions.append("softmax")
    # action_cell_counts.append(len(softmax_data))
    perform_bell(
        tf.gather(action_states, np.array(tuple(data[1] for data in bell_data), dtype=np.int32)),
        tuple(data[0] for data in bell_data)
        )
    # times.append(time.time())
    # actions.append("bell")
    # action_cell_counts.append(len(bell_data))
    perform_einsum(
        tf.gather(action_states, np.array(tuple(data[1] for data in einsum_data), dtype=np.int32)),
        tuple(data[0] for data in einsum_data),
        fields
    )
    # times.append(time.time())
    # actions.append("einsum")
    # action_cell_counts.append(len(einsum_data))
    perform_conv(
        tf.gather(action_states, np.array(tuple(data[1] for data in conv_data), dtype=np.int32)),
        tuple(data[0] for data in conv_data),
        fields
    )
    # times.append(time.time())
    # actions.append("conv")
    # action_cell_counts.append(len(conv_data))
    new_positions, new_distance_scalings = perform_move(
        tf.gather(action_states, np.array(tuple(data[1] for data in move_data), dtype=np.int32)),
        tuple(data[0] for data in move_data),
        cell_positions,
        move_gather_arg,
        move_updating_indexes,
        cell_distance_scalings,
        move_scaling_gather_indices
    )
    # times.append(time.time())
    # actions.append("move")
    # action_cell_counts.append(len(move_data))
    perform_mate(
        tf.gather(action_states, np.array(tuple(data[1] for data in mate_data), dtype=np.int32)),
        tuple(data[0] for data in mate_data),
        mating_list
    )
    # times.append(time.time())
    # actions.append("mate")
    # action_cell_counts.append(len(mate_data))
    perform_add_epigene(
        tf.gather(action_states, np.array(tuple(data[1] for data in add_epigene_data), dtype=np.int32)),
        tuple(data[0] for data in add_epigene_data)
    )
    # times.append(time.time())
    # actions.append("add_epigene")
    # action_cell_counts.append(len(add_epigene_data))
    perform_remove_epigene(
        tf.gather(action_states, np.array(tuple(data[1] for data in remove_epigene_data), dtype=np.int32)),
        tuple(data[0] for data in remove_epigene_data)
    )
    # times.append(time.time())
    # actions.append("remove_epigene")
    # action_cell_counts.append(len(remove_epigene_data))
    for cell_lock_data in lock_data:
        cell_lock_data[0].lock()
    for cell_lock_data in unlock_data:
        cell_lock_data[0].unlock()
    # times.append(time.time())
    # actions.append("lock_unlock")
    # action_cell_counts.append(len(lock_data) + len(unlock_data))
    perform_transfer_energy(
        tf.gather(action_states, np.array(tuple(data[1] for data in transfer_energy_data), dtype=np.int32)),
        tuple(data[0] for data in transfer_energy_data),
        transfer_list
    )
    # times.append(time.time())
    # actions.append("transfer")
    # action_cell_counts.append(len(transfer_energy_data))
    perform_concat(
        tf.gather(action_states, np.array(tuple(data[1] for data in concat_data), dtype=np.int32)),
        tuple(data[0] for data in concat_data),
        fields
    )
    # times.append(time.time())
    # actions.append("concat")
    # action_cell_counts.append(len(concat_data))
    perform_multiply(
        tf.gather(action_states, np.array(tuple(data[1] for data in multiply_data), dtype=np.int32)),
        tuple(data[0] for data in multiply_data),
        fields
    )
    # times.append(time.time())
    # actions.append("divide")
    # action_cell_counts.append(len(divide_data))
    perform_add(
        tf.gather(action_states, np.array(tuple(data[1] for data in add_data), dtype=np.int32)),
        tuple(data[0] for data in add_data),
        fields
    )
    # times.append(time.time())
    # actions.append("add")
    # action_cell_counts.append(len(add_data))
    output_signal = perform_signal(cells, new_hidden, new_positions, new_distance_scalings)
    # times.append(time.time())
    # actions.append("signal")
    # action_cell_counts.append(len(cells))
    # time_segments = [f"{action}: {time_a - time_b}; avg: {(time_a - time_b) / float(count)}" for time_a, time_b, action, count in zip(times[1:], times[:-1], actions, action_cell_counts) if count > 0]
    # total = times[-1] - times[0]
    return new_hidden, new_positions, output_signal, new_distance_scalings


