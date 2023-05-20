import tensorflow as tf
import numpy as np
from Cell import ChromosomeSet, ActionSet
from typing import Tuple
import Field

CELL_CENTRAL_LAYER_COUNT = 8
CELL_CENTRAL_LAYER_SIZE = 32

def generate_cell_context(cell_field_selections, fields_values, fields_cell_receptors, fields_receptor_ops, cell_energy, cell_last_reward):
    fields_receptor_outputs = []
    for values, cell_receptors, receptor_ops in zip(fields_values, fields_cell_receptors, fields_receptor_ops):
        fields_receptor_outputs.append(receptor_ops(cell_receptors, values))
    receptor_outputs = tf.reduce_sum(fields_receptor_outputs, axis=0)
    return tf.stack([cell_field_selections, receptor_outputs, cell_energy, cell_last_reward], axis=1)

def batch_genes(cells, chromosome : ChromosomeSet, chromosome_size):
    cell_genes = (cell.provide_chromosome(chromosome) for cell in cells)
    for gene_index in range(chromosome_size):
        yield tf.stack((gene[gene_index] for gene in cell_genes))


def update_hidden(hidden_states, context_hints, cells, cell_epigenetics):
    state_0 = tf.stack((hidden_states, context_hints), axis=1)
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
    w1, b1, w2, b2 = batch_genes(cells, ChromosomeSet.SELECT, 4)
    state_1 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1))
    selection_heat = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w2, state_1) + b2))
    masked_selection_heats = tf.multiply(selection_heat, cell_action_filters)
    return tf.argmax(masked_selection_heats, axis=1)


def perform_reset(hidden_state_list, cells):
    hidden_states = tf.stack(hidden_state_list, axis=0)
    cell_genes = (cell.provide_chromosome(ChromosomeSet.RESET) for cell in cells)
    w1 = tf.stack((reset_gene[0] for reset_gene in cell_genes))
    b1 = tf.stack((reset_gene[1] for reset_gene in cell_genes))
    w2 = tf.stack((reset_gene[2] for reset_gene in cell_genes))
    b2 = tf.stack((reset_gene[3] for reset_gene in cell_genes))
    state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1)
    state_2 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w2, state_1) + b2))
    for cell, field_selection in zip(cells, tf.argmax(state_2, axis=1)):
        cell.reset(field_selection)


def perform_field_shift(hidden_state_list, cells, fields : Tuple[Field]):
    hidden_states = tf.stack(hidden_state_list, axis=0)
    cell_genes = (cell.provide_chromosome(ChromosomeSet.FIELD_SHIFT) for cell in cells)
    cell_field_shift_allowances = tf.stack((fields[cell.field_index].field_shift_allowances for cell in cells))
    w1 = tf.stack((reset_gene[0] for reset_gene in cell_genes))
    b1 = tf.stack((reset_gene[1] for reset_gene in cell_genes))
    w2 = tf.stack((reset_gene[2] for reset_gene in cell_genes))
    b2 = tf.stack((reset_gene[3] for reset_gene in cell_genes))
    state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1)
    state_2 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w2, state_1) + b2))
    field_shift_heat_map = tf.multiply(state_2, cell_field_shift_allowances)
    for cell, field_selection in zip(cells, tf.argmax(field_shift_heat_map, axis=1)):
        cell.add_field_shift(field_selection)


def perform_projection(hidden_state_list, cells):
    hidden_states = tf.stack(hidden_state_list, axis=0)
    cell_projection_allowances = tf.stack((cell.current_output_field.projection_allowances for cell in cells))
    cell_genes = (cell.provide_chromosome(ChromosomeSet.PROJECTION) for cell in cells)
    w1 = tf.stack((reset_gene[0] for reset_gene in cell_genes))
    b1 = tf.stack((reset_gene[1] for reset_gene in cell_genes))
    w2 = tf.stack((reset_gene[2] for reset_gene in cell_genes))
    b2 = tf.stack((reset_gene[3] for reset_gene in cell_genes))
    w3 = tf.stack((reset_gene[4] for reset_gene in cell_genes))
    b3 = tf.stack((reset_gene[5] for reset_gene in cell_genes))
    state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1)
    state_2 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w2, state_1) + b2))
    state_3 = tf.matmul(w3, state_2) + b3
    perform_projection_heat_map = tf.multiply(state_3, cell_projection_allowances)
    for cell, field_selection, key in zip(cells, tf.argmax(perform_projection_heat_map, axis=1), state_2):
        cell.add_projection(field_selection, key)


"""
def perform_softmax(hidden_states, cell_genes):
    state_1 = tf.nn.relu(tf.matmul(cell_genes[0][0], hidden_states) + cell_genes[0][1])
    state_2 = tf.math.atan(tf.nn.relu(tf.matmul(cell_genes[1][0], state_1) + cell_genes[1][1]))
    pass
"""


def perform_einsum(hidden_state_list, cells):
    hidden_states = tf.stack(hidden_state_list, axis=0)
    cell_einsum_allowances = tf.stack((cell.current_output_field.einsum_allowances for cell in cells))
    cell_genes = (cell.provide_chromosome(ChromosomeSet.EINSUM) for cell in cells)
    w1 = tf.stack((reset_gene[0] for reset_gene in cell_genes))
    b1 = tf.stack((reset_gene[1] for reset_gene in cell_genes))
    w2 = tf.stack((reset_gene[2] for reset_gene in cell_genes))
    b2 = tf.stack((reset_gene[3] for reset_gene in cell_genes))
    w3 = tf.stack((reset_gene[4] for reset_gene in cell_genes))
    b3 = tf.stack((reset_gene[5] for reset_gene in cell_genes))
    state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1)
    state_2 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w2, state_1) + b2))
    state_3 = tf.einsum('cik,ci->ck', w3, state_2) + b3
    field_with_selection_source, field_to_selection_source = tf.split(state_3, 2)
    field_shift_heat_map_with = tf.multiply(field_with_selection_source, cell_einsum_allowances)
    for cell, field_selection_with in zip(cells, tf.argmax(field_shift_heat_map_with, axis=1)):
        cell_einsum_allowances_with = cell.current_output_field.einsum_allowances_with[field_selection_with]
        field_shift_heat_map_to = tf.multiply(field_to_selection_source, cell_einsum_allowances_with)
        field_selection_to = tf.argmax(field_shift_heat_map_to, axis=1)
        cell.add_einsum(field_selection_with, field_selection_to)


def perform_conv(hidden_state_list, cells, fields):
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
    # Consolidate the hidden states provided for each cell into a single tensor to be operated on in a single operation
    hidden_states = tf.stack(hidden_state_list, axis=0)
    # conv_allowances indicates which other fields a given field can output to using a convolution operation
    field_conv_selection_mask = tf.stack((fields[cell.field_index].conv_allowances for cell in cells))
    # Get the parameters used in the instructions to add convolution operations to organisms
    w1, b1, w2, b2, w3, b3 = \
        batch_genes(cells, ChromosomeSet.CONV, 6)
    # Compute state_1 by applying a ReLU activation on an einsum between w1 and hidden_states added to b1
    state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1)
    # Compute state_2 by applying an atan activation on the ReLU activation of an einsum between  w2 and state_1 added to b2
    state_2 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w2, state_1) + b2))
    # Compute state_3 by taking an einsum between w3 and state_2 and adding b3
    state_3 = tf.einsum('cik,ci->ck', w3, state_2) + b3
    # Multiply state_3 by the field_conv_selection_mask to create a tensor of values indicating the preference of each organism to generate an output to each field
    conv_heat_map = tf.multiply(state_3, field_conv_selection_mask)
    # Loop through each cell, field_selection, and kernel_generation_seed
    for cell, field_selection, kernel_generation_seed in zip(cells, tf.argmax(conv_heat_map, axis=1), state_2):
        # Add a convolution layer to the organism, which will output to the field indicated by field_selection and
        # use weights and biases generated pseudorandomly using the value in kernel_generation_seed
        # The cell's field_index property will be updated to the value of field_selection
        cell.add_conv(field_selection, kernel_generation_seed)


def perform_move(hidden_state_list, cells):
    hidden_states = tf.stack(hidden_state_list, axis=0)
    cell_genes = (cell.provide_chromosome(ChromosomeSet.MOVE) for cell in cells)
    w1 = tf.stack((reset_gene[0] for reset_gene in cell_genes))
    b1 = tf.stack((reset_gene[1] for reset_gene in cell_genes))
    w2 = tf.stack((reset_gene[2] for reset_gene in cell_genes))
    b2 = tf.stack((reset_gene[3] for reset_gene in cell_genes))
    state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1)
    state_2 = tf.einsum('cik,ci->ck', w2, state_1) + b2
    for cell, direction in zip(cells, state_2):
        cell.move(direction[0], direction[1])


def perform_mate(hidden_state_list, cells, mating_list):
    hidden_states = tf.stack(hidden_state_list, axis=0)
    cell_genes = (cell.provide_chromosome(ChromosomeSet.MATE) for cell in cells)
    w1 = tf.stack((reset_gene[0] for reset_gene in cell_genes))
    b1 = tf.stack((reset_gene[1] for reset_gene in cell_genes))
    w2 = tf.stack((reset_gene[2] for reset_gene in cell_genes))
    b2 = tf.stack((reset_gene[3] for reset_gene in cell_genes))
    state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1)
    state_2 = tf.einsum('cik,ci->ck', w2, state_1) + b2
    for cell, direction in zip(cells, state_2):
        if cell.mate():
            mating_list.append((cell, direction))


def perform_add_epigene(hidden_state_list, cells):
    hidden_states = tf.stack(hidden_state_list, axis=0)
    cell_genes = (cell.provide_chromosome(ChromosomeSet.MATE) for cell in cells)
    w1 = tf.stack((reset_gene[0] for reset_gene in cell_genes))
    b1 = tf.stack((reset_gene[1] for reset_gene in cell_genes))
    w2 = tf.stack((reset_gene[2] for reset_gene in cell_genes))
    b2 = tf.stack((reset_gene[3] for reset_gene in cell_genes))
    state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1)
    state_2 = tf.einsum('cik,ci->ck', w2, state_1) + b2
    layer_select_heat_map, index_select_heat_map = tf.split(state_2, [CELL_CENTRAL_LAYER_COUNT, CELL_CENTRAL_LAYER_SIZE])
    cells_layer_selection = tf.argmax(layer_select_heat_map, axis=1)
    cells_index_selection = tf.argmax(index_select_heat_map, axis=1)
    for cell, layer_selection, index_selection in zip(cells, cells_layer_selection, cells_index_selection):
        cell.add_epigene(layer_selection, index_selection)


def perform_remove_epigene(hidden_state_list, cells):
    hidden_states = tf.stack(hidden_state_list, axis=0)
    cell_genes = (cell.provide_chromosome(ChromosomeSet.MATE) for cell in cells)
    w1 = tf.stack((reset_gene[0] for reset_gene in cell_genes))
    b1 = tf.stack((reset_gene[1] for reset_gene in cell_genes))
    w2 = tf.stack((reset_gene[2] for reset_gene in cell_genes))
    b2 = tf.stack((reset_gene[3] for reset_gene in cell_genes))
    state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1)
    state_2 = tf.einsum('cik,ci->ck', w2, state_1) + b2
    layer_select_heat_map, cells_index_select_heat_map = tf.split(state_2, [CELL_CENTRAL_LAYER_COUNT, CELL_CENTRAL_LAYER_SIZE])
    cells_layer_selection = tf.argmax(layer_select_heat_map, axis=1)
    for cell, layer_selection, index_heat_map in zip(cells, cells_layer_selection, cells_index_select_heat_map):
        cell.subtract_epigene(layer_selection, index_heat_map)


def perform_transfer_energy(hidden_state_list, cells, transfer_list):
    hidden_states = tf.stack(hidden_state_list, axis=0)
    cell_genes = (cell.provide_chromosome(ChromosomeSet.MATE) for cell in cells)
    w1 = tf.stack((reset_gene[0] for reset_gene in cell_genes))
    b1 = tf.stack((reset_gene[1] for reset_gene in cell_genes))
    w2 = tf.stack((reset_gene[2] for reset_gene in cell_genes))
    b2 = tf.stack((reset_gene[3] for reset_gene in cell_genes))
    state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1)
    state_2 = tf.einsum('cik,ci->ck', w2, state_1) + b2
    cells_direction, cells_key = tf.split(state_2, [4, 4])
    for cell, direction, key in zip(cells, cells_direction, cells_key):
        transfer_list.append((cell, direction, key))


def perform_concat(hidden_state_list, cells, fields):
    hidden_states = tf.stack(hidden_state_list, axis=0)
    cell_genes = (cell.provide_chromosome(ChromosomeSet.CONCAT) for cell in cells)
    cell_field_shift_allowances = tf.stack((fields[cell.field_index].concat_allowances for cell in cells))
    w1 = tf.stack((concat_gene[0] for concat_gene in cell_genes))
    b1 = tf.stack((concat_gene[1] for concat_gene in cell_genes))
    w2 = tf.stack((concat_gene[2] for concat_gene in cell_genes))
    b2 = tf.stack((concat_gene[3] for concat_gene in cell_genes))
    state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1)
    state_2 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w2, state_1) + b2))
    concat_heat_map = tf.multiply(state_2, cell_field_shift_allowances)
    for cell, field_selection in zip(cells, tf.argmax(concat_heat_map, axis=1)):
        cell.add_concat(field_selection)

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
    # Consolidate the hidden states provided for each cell into a single tensor to be operated on in a single operation
    hidden_states = tf.stack(hidden_state_list, axis=0)
    # conv_allowances indicates which other fields a given field can output to using a convolution operation
    divide_selection_mask = tf.stack((fields[cell.field_index].divide_allowances for cell in cells))
    # Get the parameters used in the instructions to add convolution operations to organisms
    w1, b1, w2, b2, w3, b3 = \
        batch_genes(cells, ChromosomeSet.DIVIDE, 6)
    # Compute state_1 by applying a ReLU activation on an einsum between w1 and hidden_states added to b1
    state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1)
    # Compute state_2 by applying an atan activation on the ReLU activation of an einsum between  w2 and state_1 added to b2
    state_2 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w2, state_1) + b2))
    # Compute state_3 by taking an einsum between w3 and state_2 and adding b3
    state_3 = tf.einsum('cik,ci->ck', w3, state_2) + b3
    # Multiply state_3 by the field_conv_selection_mask to create a tensor of values indicating the preference of each organism to generate an output to each field
    divide_heat_map = tf.multiply(state_3, divide_selection_mask)
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
    # Consolidate the hidden states provided for each cell into a single tensor to be operated on in a single operation
    hidden_states = tf.stack(hidden_state_list, axis=0)
    # conv_allowances indicates which other fields a given field can output to using a convolution operation
    add_selection_mask = tf.stack((fields[cell.field_index].add_allowances for cell in cells))
    # Get the parameters used in the instructions to add convolution operations to organisms
    w1, b1, w2, b2, w3, b3 = \
        batch_genes(cells, ChromosomeSet.ADD, 6)
    # Compute state_1 by applying a ReLU activation on an einsum between w1 and hidden_states added to b1
    state_1 = tf.nn.relu(tf.einsum('cik,ci->ck', w1, hidden_states) + b1)
    # Compute state_2 by applying an atan activation on the ReLU activation of an einsum between  w2 and state_1 added to b2
    state_2 = tf.math.atan(tf.nn.relu(tf.einsum('cik,ci->ck', w2, state_1) + b2))
    # Compute state_3 by taking an einsum between w3 and state_2 and adding b3
    state_3 = tf.einsum('cik,ci->ck', w3, state_2) + b3
    # Multiply state_3 by the field_conv_selection_mask to create a tensor of values indicating the preference of each organism to generate an output to each field
    add_heat_map = tf.multiply(state_3, add_selection_mask)
    # Loop through each cell, field_selection, and kernel_generation_seed
    for cell, field_selection in zip(cells, tf.argmax(add_heat_map, axis=1)):
        # Add a convolution layer to the organism, which will output to the field indicated by field_selection and
        # use weights and biases generated pseudorandomly using the value in kernel_generation_seed
        # The cell's field_index property will be updated to the value of field_selection
        cell.add_divide(field_selection)

def operate(cells, fields : Tuple[Field], cell_receptors, cell_hidden_states, mating_list, transfer_list):
    field_selections = tf.TensorArray(dtype=tf.float32, size=len(cells))
    cell_action_filters = tf.TensorArray(dtype=tf.float32, size=len(cells))
    cell_energy = tf.TensorArray(dtype=tf.float32, size=len(cells))
    cell_reward = tf.TensorArray(dtype=tf.float32, size=len(cells))
    for cell_index, cell in enumerate(cells):
        field_selection = np.zeros(len(fields))
        field_selection[cell.field_index] = 1
        field_selections = field_selections.write(cell_index, field_selection)
        cell_action_filters = cell_action_filters.write(cell_index, cell.action_mask)
        cell_energy = cell_energy.write(cell_index, cell.energy)
        cell_reward = cell_reward.write(cell_index, cell.last_reward)
    context_hints = generate_cell_context(
        field_selections.stack(),
        [field.value for field in fields],
        cell_receptors,
        [field.receptor_ops for field in fields],
        cell_energy.stack(),
        cell_reward.stack()
    )
    cell_epigenetics = []
    for epigene_index in range(CELL_CENTRAL_LAYER_COUNT):
        epigene = tf.TensorArray(dtype=tf.float32, size=len(cells))
        for cell_index, cell in enumerate(cells):
            epigene = epigene.write(cell_index, cell.provide_epigenes(epigene_index))
        cell_epigenetics.append(epigene.stack())
    new_hidden = update_hidden(cell_hidden_states, context_hints, cells, cell_epigenetics)
    cell_selections = select_action(new_hidden, cells, cell_action_filters)
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
    for cell, selection, cell_new_hidden in zip(cells, cell_selections, new_hidden):
        data_set[selection].append((cell, cell_new_hidden))
    perform_reset(
        tf.stack((data[1] for data in reset_data)),
        (data[0] for data in reset_data)
    )
    perform_field_shift(
        tf.stack((data[1] for data in field_shift_data)),
        (data[0] for data in field_shift_data),
        fields
    )
    perform_projection(
        tf.stack((data[1] for data in projection_data)),
        (data[0] for data in projection_data)
    )
    for cell_lock_data in softmax_data:
        cell_lock_data[0].add_softmax()
    perform_einsum(
        tf.stack((data[1] for data in einsum_data)),
        (data[0] for data in einsum_data)
    )
    perform_conv(
        tf.stack((data[1] for data in conv_data)),
        (data[0] for data in conv_data),
        fields
    )
    perform_move(
        tf.stack((data[1] for data in move_data)),
        (data[0] for data in move_data)
    )
    perform_mate(
        tf.stack((data[1] for data in mate_data)),
        (data[0] for data in mate_data),
        mating_list
    )
    perform_add_epigene(
        tf.stack((data[1] for data in add_epigene_data)),
        (data[0] for data in add_epigene_data)
    )
    perform_remove_epigene(
        tf.stack((data[1] for data in remove_epigene_data)),
        (data[0] for data in remove_epigene_data)
    )
    for cell_lock_data in lock_data:
        cell_lock_data[0].lock()
    for cell_lock_data in unlock_data:
        cell_lock_data[0].unlock()
    perform_transfer_energy(
        tf.stack((data[1] for data in transfer_energy_data)),
        (data[0] for data in transfer_energy_data),
        transfer_list
    )
    perform_concat(
        tf.stack((data[1] for data in concat_data)),
        (data[0] for data in concat_data),
        fields
    )
    perform_divide(
        tf.stack((data[1] for data in divide_data)),
        (data[0] for data in concat_data),
        fields
    )
    perform_add(
        tf.stack((data[1] for data in add_data)),
        (data[0] for data in concat_data),
        fields
    )
    return new_hidden

