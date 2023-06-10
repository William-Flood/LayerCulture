import tensorflow as tf
import numpy as np
from Cell import ChromosomeSet, ActionSet
from typing import Tuple
from Field import Field

CELL_CENTRAL_LAYER_COUNT = 8
CELL_CENTRAL_LAYER_SIZE = 32
GENERATION_KEY_SIZE = 8

def generate_cell_context(cell_field_selections, cell_signal_field_values, cells, cell_energy, cell_last_reward, cell_transmit_reward, cell_receive_reward):
    w_1, b_1, w_2, b_2 = \
        batch_genes(cells, ChromosomeSet.RECEPTORS, 4)
    receptor_hidden = tf.nn.relu(tf.einsum("cds,cdso->co", cell_signal_field_values, w_1) + b_1)
    receptor_outputs = tf.einsum("cv,cvo->co", receptor_hidden, w_2) + b_2
    return tf.concat([cell_field_selections, receptor_outputs, cell_energy, cell_last_reward, cell_transmit_reward, cell_receive_reward], axis=1)

def batch_genes(cells, chromosome : ChromosomeSet, chromosome_size):
    cell_genes = tuple(cell.provide_chromosome(chromosome) for cell in cells)
    for gene_index in range(chromosome_size):
        yield tf.stack(tuple(gene[gene_index] for gene in cell_genes))


def update_hidden(hidden_states, context_hints, cells, cell_epigenetics):
    state_0 = tf.concat((hidden_states, context_hints), axis=1)
    w_1, b_1, w_2, b_2, w_3, b_3, w_4, b_4, w_5, b_5, w_6, b_6, w_7, b_7, w_8, b_8 = \
        batch_genes(cells, ChromosomeSet.MAIN, 16)
    state_1 = tf.nn.relu(tf.multiply(tf.einsum('cik,ci->ck', w_1, state_0) + b_1, cell_epigenetics[0]))
    state_2 = tf.math.atan(
        tf.nn.relu(tf.multiply(tf.einsum('cik,ci->ck', w_2, state_1) + b_2, cell_epigenetics[1])))
    state_3 = tf.nn.relu(tf.multiply(tf.einsum('cik,ci->ck', w_3, state_2) + b_3, cell_epigenetics[2]))
    state_4 = tf.math.atan(
        tf.nn.relu(tf.multiply(tf.einsum('cik,ci->ck', w_4, state_3) + b_4, cell_epigenetics[3])))
    state_5 = tf.nn.relu(tf.multiply(tf.einsum('cik,ci->ck', w_5, state_4) + b_5, cell_epigenetics[4]))
    state_6 = tf.math.atan(
        tf.nn.relu(tf.multiply(tf.einsum('cik,ci->ck', w_6, state_5) + b_6, cell_epigenetics[5])))
    state_7 = tf.nn.relu(tf.multiply(tf.einsum('cik,ci->ck', w_7, state_6) + b_7, cell_epigenetics[6]))
    state_8 = tf.math.atan(tf.multiply(tf.einsum('cik,ci->ck', w_8, state_7) + b_8, cell_epigenetics[7]))
    return state_8


def select_action(hidden_states, cells, cell_action_filters):
    w1, b1, w2, b2, w3, b3 = batch_genes(cells, ChromosomeSet.SELECT, 6)
    state_1 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1))
    state_2 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w2, state_1) + b2))
    selection_heat = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w3, state_2) + b3))
    masked_selection_heats = tf.multiply(selection_heat, cell_action_filters)
    return tf.argmax(masked_selection_heats, axis=1)


def perform_reset(hidden_state_list, cells):
    if 0 < len(hidden_state_list):
        hidden_states = tf.stack(hidden_state_list, axis=0)
        w1, b1, w2, b2 = batch_genes(cells, ChromosomeSet.RESET, 4)
        state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1)
        state_2 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w2, state_1) + b2))
        for cell, field_selection in zip(cells, tf.argmax(state_2, axis=1)):
            cell.reset(field_selection)


def perform_field_shift(hidden_state_list, cells, fields: Tuple[Field]):
    if 0 < len(hidden_state_list):
        hidden_states = tf.stack(hidden_state_list, axis=0)
        cell_field_shift_allowances = tf.stack(tuple(fields[cell.field_index].field_shift_allowances for cell in cells))
        w1, b1, w2, b2 = batch_genes(cells, ChromosomeSet.FIELD_SHIFT, 4)
        state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1)
        state_2 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w2, state_1) + b2))
        normalized_field_shift_values = tf.math.sigmoid(state_2) + .01
        field_shift_heat_map = tf.multiply(normalized_field_shift_values, cell_field_shift_allowances)
        for cell, field_selection in zip(cells, tf.argmax(field_shift_heat_map, axis=1)):
            cell.add_field_shift(field_selection)


def perform_projection(hidden_state_list, cells, field_count):
    if 0 < len(hidden_state_list):
        hidden_states = tf.stack(hidden_state_list, axis=0)
        cell_projection_allowances = tf.stack(tuple(cell.current_output_field.projection_allowances for cell in cells))
        w1, b1, w2, b2, w3, b3 = batch_genes(cells, ChromosomeSet.PROJECTION, 6)
        state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1)
        state_2 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w2, state_1) + b2))
        state_3 = tf.einsum('cik,ci->ck', w3, state_2) + b3
        projection_values, keys = tf.split(state_3, [field_count, 8], axis=1)
        normalized_projection_values = tf.math.sigmoid(projection_values) + .01
        perform_projection_heat_map = tf.multiply(normalized_projection_values, cell_projection_allowances)
        for cell, field_selection, cell_key in zip(cells, tf.argmax(perform_projection_heat_map, axis=1), keys):
            cell.add_projection(field_selection, cell_key)


"""
def perform_softmax(hidden_states, cell_genes):
    state_1 = tf.nn.relu(tf.matmul(cell_genes[0][0], hidden_states) + cell_genes[0][1])
    state_2 = tf.math.atan(tf.nn.relu(tf.matmul(cell_genes[1][0], state_1) + cell_genes[1][1]))
    pass
"""


def perform_einsum(hidden_state_list, cells):
    if 0 < len(hidden_state_list):
        hidden_states = tf.stack(hidden_state_list, axis=0)
        cell_einsum_allowances = tf.stack(tuple(cell.current_output_field.einsum_allowances for cell in cells))
        w1, b1, w2, b2, w3, b3 = batch_genes(cells, ChromosomeSet.EINSUM, 6)
        state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1)
        state_2 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w2, state_1) + b2))
        state_3 = tf.einsum('cik,ci->ck', w3, state_2) + b3
        field_with_selection_source, field_to_selection_source = tf.split(state_3, 2, axis=1)
        normalized_einsum_with_values = tf.math.sigmoid(field_with_selection_source) + .01
        einsum_heat_map_with = tf.multiply(normalized_einsum_with_values, cell_einsum_allowances)
        normalized_einsum_to_values = tf.math.sigmoid(field_to_selection_source) + .01
        for cell, field_selection_with, cell_normalized_einsum_to_values in zip(cells, tf.argmax(einsum_heat_map_with, axis=1), normalized_einsum_to_values):
            cell_einsum_allowances_with = cell.current_output_field.einsum_allowances_with[int(field_selection_with)]
            field_shift_heat_map_to = tf.multiply(cell_normalized_einsum_to_values, cell_einsum_allowances_with)
            field_selection_to = tf.argmax(field_shift_heat_map_to)
            cell.add_einsum(field_selection_with, field_selection_to)


def perform_conv(hidden_state_list, cells, field_count):
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
    if 0 < len(hidden_state_list):
        # Consolidate the hidden states provided for each cell into a single tensor to be operated on in a single operation
        hidden_states = tf.stack(hidden_state_list, axis=0)
        # conv_allowances indicates which other fields a given field can output to using a convolution operation
        field_conv_selection_mask = tf.stack(tuple(cell.current_output_field.conv_allowances for cell in cells))
        # Get the parameters used in the instructions to add convolution operations to organisms
        w1, b1, w2, b2, w3, b3 = \
            batch_genes(cells, ChromosomeSet.CONV, 6)
        # Compute state_1 by applying a ReLU activation on an einsum between w1 and hidden_states added to b1
        state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1)
        # Compute state_2 by applying an atan activation on the ReLU activation of an einsum between  w2 and state_1 added to b2
        state_2 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w2, state_1) + b2))
        # Compute state_3 by taking an einsum between w3 and state_2 and adding b3
        state_3 = tf.einsum('cik,ci->ck', w3, state_2) + b3
        # Break out the components of the chromosome evaluation for their separate roles in building the cell graph
        field_with_selection_source, kernel_generation_seed = tf.split(state_3, [field_count, 2 * GENERATION_KEY_SIZE], axis=1)
        # Ensure that all values in the heat map used to select the target field of the convolution operation are positive
        # This ensures that the selected field is contained in field_conv_selection_mask
        normalized_field_with_selection_source = tf.math.sigmoid(field_with_selection_source) + .01
        # Multiply state_3 by the field_conv_selection_mask to create a tensor of values indicating the preference of each organism to generate an output to each field
        conv_heat_map = tf.multiply(normalized_field_with_selection_source, field_conv_selection_mask)
        # Loop through each cell, field_selection, and kernel_generation_seed
        for cell, field_selection, cell_kernel_generation_seed in zip(cells, tf.argmax(conv_heat_map, axis=1), kernel_generation_seed):
            # Add a convolution layer to the organism, which will output to the field indicated by field_selection and
            # use weights and biases generated pseudorandomly using the value in kernel_generation_seed
            # The cell's field_index property will be updated to the value of field_selection
            cell.add_conv(field_selection, cell_kernel_generation_seed)


def perform_move(hidden_state_list, cells, origial_positions, move_gather_arg):
    if 0 < len(hidden_state_list):
        hidden_states = tf.stack(hidden_state_list, axis=0)
        w1, b1, w2, b2 = batch_genes(cells, ChromosomeSet.MOVE, 4)
        state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1)
        state_2 = tf.einsum('cik,ci->ck', w2, state_1) + b2
        for cell, direction in zip(cells, state_2):
            cell.move(direction[0], direction[1], direction[2], direction[3])

        static_and_deltas = tf.concat([tf.zeros([1, 4]), state_2], axis=0)
        new_positions = origial_positions + tf.gather(static_and_deltas, move_gather_arg, axis=0)
        return new_positions
    else:
        return origial_positions


def perform_mate(hidden_state_list, cells, mating_list):
    if 0 < len(hidden_state_list):
        hidden_states = tf.stack(hidden_state_list, axis=0)
        w1, b1, w2, b2 = batch_genes(cells, ChromosomeSet.MATE, 4)
        state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1)
        state_2 = tf.einsum('cik,ci->ck', w2, state_1) + b2
        for cell, direction in zip(cells, state_2):
            if cell.mate():
                mating_list.append((cell, direction))


def perform_add_epigene(hidden_state_list, cells):
    if 0 < len(hidden_state_list):
        hidden_states = tf.stack(hidden_state_list, axis=0)
        w1, b1, w2, b2 = batch_genes(cells, ChromosomeSet.ADD_EPIGENE, 4)
        state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1)
        state_2 = tf.einsum('cik,ci->ck', w2, state_1) + b2
        layer_select_heat_map, index_select_heat_map = tf.split(state_2, [CELL_CENTRAL_LAYER_COUNT, CELL_CENTRAL_LAYER_SIZE], axis=1)
        cells_layer_selection = tf.argmax(layer_select_heat_map, axis=1)
        cells_index_selection = tf.argmax(index_select_heat_map, axis=1)
        for cell, layer_selection, index_selection in zip(cells, cells_layer_selection, cells_index_selection):
            cell.add_epigene(layer_selection, index_selection)


def perform_remove_epigene(hidden_state_list, cells):
    if 0 < len(hidden_state_list):
        hidden_states = tf.stack(hidden_state_list, axis=0)
        w1, b1, w2, b2 = batch_genes(cells, ChromosomeSet.REMOVE_EPIGENE, 4)
        state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1)
        state_2 = tf.einsum('cik,ci->ck', w2, state_1) + b2
        layer_select_heat_map, cells_index_select_heat_map = tf.split(state_2, [CELL_CENTRAL_LAYER_COUNT, CELL_CENTRAL_LAYER_SIZE], axis=1)
        cells_layer_selection = tf.argmax(layer_select_heat_map, axis=1)
        for cell, layer_selection, index_heat_map in zip(cells, cells_layer_selection, cells_index_select_heat_map):
            cell.subtract_epigene(layer_selection, index_heat_map)


def perform_transfer_energy(hidden_state_list, cells, transfer_list):
    if 0 < len(hidden_state_list):
        hidden_states = tf.stack(hidden_state_list, axis=0)
        w1, b1, w2, b2 = batch_genes(cells, ChromosomeSet.TRANSFER_ENERGY, 4)
        state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1)
        state_2 = tf.einsum('cik,ci->ck', w2, state_1) + b2
        cells_direction, cells_key = tf.split(state_2, [4, 4], axis=1)
        for cell, direction, key in zip(cells, cells_direction, cells_key):
            transfer_list.append((cell, direction, key))


def perform_concat(hidden_state_list, cells):
    if 0 < len(hidden_state_list):
        hidden_states = tf.stack(hidden_state_list, axis=0)
        cell_field_shift_allowances = tf.stack(tuple(cell.current_output_field.concat_allowances for cell in cells))
        w1, b1, w2, b2, w3, b3 = batch_genes(cells, ChromosomeSet.CONCAT, 6)
        state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1)
        state_2 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w2, state_1) + b2))
        state_3 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w3, state_2) + b3))
        field_with_selection_source, field_to_selection_source = tf.split(state_3, 2, axis=1)
        normalized_field_with_selection_source = tf.math.sigmoid(field_with_selection_source) + .01
        concat_heat_map = tf.multiply(normalized_field_with_selection_source, cell_field_shift_allowances)
        for cell, field_selection, cell_field_to_selection_source in zip(cells, tf.argmax(concat_heat_map, axis=1), field_to_selection_source):
            to_allowances = cell.current_output_field.concat_to_allowances(int(field_selection))
            concat_to_heat_map = tf.multiply(cell_field_to_selection_source, to_allowances)
            to_selection = tf.argmax(concat_to_heat_map)
            cell.add_concat(field_selection, to_selection)

def perform_divide(hidden_state_list, cells, fields):
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
        # Consolidate the hidden states provided for each cell into a single tensor to be operated on in a single operation
        hidden_states = tf.stack(hidden_state_list, axis=0)
        # conv_allowances indicates which other fields a given field can output to using a convolution operation
        divide_selection_mask = tf.stack(tuple(fields[cell.field_index].elementwise_allowances for cell in cells))
        # Get the parameters used in the instructions to add convolution operations to organisms
        w1, b1, w2, b2, w3, b3 = \
            batch_genes(cells, ChromosomeSet.DIVIDE, 6)
        # Compute state_1 by applying a ReLU activation on an einsum between w1 and hidden_states added to b1
        state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1)
        # Compute state_2 by applying an atan activation on the ReLU activation of an einsum between  w2 and state_1 added to b2
        state_2 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w2, state_1) + b2))
        # Compute state_3 by taking an einsum between w3 and state_2 and adding b3
        state_3 = tf.einsum('cik,ci->ck', w3, state_2) + b3
        normalized_field_with_selection_source = tf.math.sigmoid(state_3) + .01
        # Multiply state_3 by the field_conv_selection_mask to create a tensor of values indicating the preference of each organism to generate an output to each field
        divide_heat_map = tf.multiply(normalized_field_with_selection_source, divide_selection_mask)
        # Loop through each cell, field_selection, and kernel_generation_seed
        for cell, field_selection in zip(cells, tf.argmax(divide_heat_map, axis=1)):
            # Add a convolution layer to the organism, which will output to the field indicated by field_selection and
            # use weights and biases generated pseudorandomly using the value in kernel_generation_seed
            # The cell's field_index property will be updated to the value of field_selection
            cell.add_divide(field_selection)

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
        # Consolidate the hidden states provided for each cell into a single tensor to be operated on in a single operation
        hidden_states = tf.stack(hidden_state_list, axis=0)
        # conv_allowances indicates which other fields a given field can output to using a convolution operation
        add_selection_mask = tf.stack(tuple(fields[cell.field_index].elementwise_allowances for cell in cells))
        # Get the parameters used in the instructions to add convolution operations to organisms
        w1, b1, w2, b2, w3, b3 = \
            batch_genes(cells, ChromosomeSet.ADD, 6)
        # Compute state_1 by applying a ReLU activation on an einsum between w1 and hidden_states added to b1
        state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1)
        # Compute state_2 by applying an atan activation on the ReLU activation of an einsum between  w2 and state_1 added to b2
        state_2 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w2, state_1) + b2))
        # Compute state_3 by taking an einsum between w3 and state_2 and adding b3
        state_3 = tf.einsum('cik,ci->ck', w3, state_2) + b3
        normalized_field_with_selection_source = tf.math.sigmoid(state_3) + .01
        # Multiply state_3 by the field_conv_selection_mask to create a tensor of values indicating the preference of each organism to generate an output to each field
        add_heat_map = tf.multiply(normalized_field_with_selection_source, add_selection_mask)
        # Loop through each cell, field_selection, and kernel_generation_seed
        for cell, field_selection in zip(cells, tf.argmax(add_heat_map, axis=1)):
            # Add a convolution layer to the organism, which will output to the field indicated by field_selection and
            # use weights and biases generated pseudorandomly using the value in kernel_generation_seed
            # The cell's field_index property will be updated to the value of field_selection
            cell.add_divide(field_selection)

def perform_signal(cells, hidden_states, cell_positions):
    if 0 < len(hidden_states):
        cell_count = len(cells)
        a_position_tensor = tf.tile(tf.reshape(cell_positions, [cell_count, 1, 4]), [1, cell_count, 1])
        b_position_tensor = tf.tile(tf.reshape(cell_positions, [1, cell_count, 4 ]), [cell_count, 1, 1])
        cell_identity = tf.eye(cell_count, dtype=tf.float32)
        displacements = a_position_tensor - b_position_tensor
        distances = tf.math.sqrt(tf.reduce_sum(tf.square(displacements), axis=2))
        cell_ones = tf.ones([cell_count, cell_count], dtype=tf.float32)
        distance_scaling = tf.divide(cell_ones, distances + cell_ones)
        masked_distances = tf.multiply(distance_scaling, cell_ones - cell_identity)
        w1, b1, w2, b2 = \
            batch_genes(cells, ChromosomeSet.EMIT, 4)
        # Compute state_1 by applying a ReLU activation on an einsum between w1 and hidden_states added to b1
        state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1)
        # Compute state_2 by applying an atan activation on the ReLU activation of an einsum between  w2 and state_1 added to b2
        state_2 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w2, state_1) + b2))
        front_mask = (displacements[:, :, 0] + 1.0) / 2.0
        back_mask = cell_ones - front_mask
        front_back_masked_distances = tf.stack(
            (
                tf.multiply(masked_distances, front_mask),
                tf.multiply(masked_distances, back_mask)
            ),
            axis=1
        )
        cell_signal_inputs = tf.einsum("cbd,dk->cbk", front_back_masked_distances, state_2)
        return cell_signal_inputs

def operate(cells, fields : Tuple[Field], cell_hidden_states, cell_signal_values, mating_list, transfer_list, cell_positions):
    field_selections = tf.TensorArray(dtype=tf.float32, size=len(cells))
    cell_action_filters = tf.TensorArray(dtype=tf.float32, size=len(cells))
    cell_energy = tf.TensorArray(dtype=tf.float32, size=len(cells))
    cell_reward = tf.TensorArray(dtype=tf.float32, size=len(cells))
    cell_transmit_reward = tf.TensorArray(dtype=tf.float32, size=len(cells))
    cell_receive_reward = tf.TensorArray(dtype=tf.float32, size=len(cells))
    for cell_index, cell in enumerate(cells):
        field_selection = np.zeros(len(fields))
        field_selection[cell.field_index] = 1
        field_selections = field_selections.write(cell_index, field_selection)
        cell_action_filters = cell_action_filters.write(cell_index, cell.action_mask)
        cell_energy = cell_energy.write(cell_index, [cell.energy])
        cell_reward = cell_reward.write(cell_index, [cell.last_reward])
        cell_transmit_reward = cell_transmit_reward.write(cell_index, [cell.last_transmit_reward])
        cell_receive_reward = cell_receive_reward.write(cell_index, [cell.last_receive_reward])
    context_hints = generate_cell_context(
        field_selections.stack(),
        cell_signal_values,
        cells,
        cell_energy.stack(),
        cell_reward.stack(),
        cell_transmit_reward.stack(),
        cell_receive_reward.stack()
    )
    cell_epigenetics = []
    for epigene_index in range(CELL_CENTRAL_LAYER_COUNT):
        epigene = tf.TensorArray(dtype=tf.float32, size=len(cells))
        for cell_index, cell in enumerate(cells):
            epigene = epigene.write(cell_index, cell.provide_epigenes(epigene_index))
        cell_epigenetics.append(epigene.stack())
    new_hidden = update_hidden(cell_hidden_states, context_hints, cells, cell_epigenetics)
    cell_selections = select_action(new_hidden, cells, cell_action_filters.stack())
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
    divide_data = []
    add_data = []
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
        divide_data,
        add_data
    )
    # wait
    move_gather_arg = np.zeros([len(cells)], dtype=np.int32)
    move_counter = 1
    for cell, selection, index in zip(cells, cell_selections, range(len(cells))):
        data_set[selection].append((cell, index))
        if(selection == int(ActionSet.MOVE)):
            move_gather_arg[index] = move_counter
            move_counter += 1
    perform_reset(
        tf.gather(new_hidden, np.array(tuple(data[1] for data in reset_data), dtype=np.int32)),
        tuple(data[0] for data in reset_data)
    )
    perform_field_shift(
        tf.gather(new_hidden, np.array(tuple(data[1] for data in field_shift_data), dtype=np.int32)),
        tuple(data[0] for data in field_shift_data),
        fields
    )
    perform_projection(
        tf.gather(new_hidden, np.array(tuple(data[1] for data in projection_data), dtype=np.int32)),
        tuple(data[0] for data in projection_data),
        len(fields)
    )
    for cell_lock_data in softmax_data:
        cell_lock_data[0].add_softmax()
    perform_einsum(
        tf.gather(new_hidden, np.array(tuple(data[1] for data in einsum_data), dtype=np.int32)),
        tuple(data[0] for data in einsum_data)
    )
    perform_conv(
        tf.gather(new_hidden, np.array(tuple(data[1] for data in conv_data), dtype=np.int32)),
        tuple(data[0] for data in conv_data),
        len(fields)
    )
    new_positions = perform_move(
        tf.gather(new_hidden, np.array(tuple(data[1] for data in move_data), dtype=np.int32)),
        tuple(data[0] for data in move_data),
        cell_positions,
        move_gather_arg
    )
    perform_mate(
        tf.gather(new_hidden, np.array(tuple(data[1] for data in mate_data), dtype=np.int32)),
        tuple(data[0] for data in mate_data),
        mating_list
    )
    perform_add_epigene(
        tf.gather(new_hidden, np.array(tuple(data[1] for data in add_epigene_data), dtype=np.int32)),
        tuple(data[0] for data in add_epigene_data)
    )
    perform_remove_epigene(
        tf.gather(new_hidden, np.array(tuple(data[1] for data in remove_epigene_data), dtype=np.int32)),
        tuple(data[0] for data in remove_epigene_data)
    )
    for cell_lock_data in lock_data:
        cell_lock_data[0].lock()
    for cell_lock_data in unlock_data:
        cell_lock_data[0].unlock()
    perform_transfer_energy(
        tf.gather(new_hidden, np.array(tuple(data[1] for data in transfer_energy_data), dtype=np.int32)),
        tuple(data[0] for data in transfer_energy_data),
        transfer_list
    )
    perform_concat(
        tf.gather(new_hidden, np.array(tuple(data[1] for data in concat_data), dtype=np.int32)),
        tuple(data[0] for data in concat_data)
    )
    perform_divide(
        tf.gather(new_hidden, np.array(tuple(data[1] for data in divide_data), dtype=np.int32)),
        tuple(data[0] for data in divide_data),
        fields
    )
    perform_add(
        tf.gather(new_hidden, np.array(tuple(data[1] for data in add_data), dtype=np.int32)),
        tuple(data[0] for data in add_data),
        fields
    )
    output_signal = perform_signal(cells, new_hidden, new_positions)
    return new_hidden, new_positions, output_signal


