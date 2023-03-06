import tensorflow as tf
import numpy as np
from Cell import ChromosomeSet, ActionSet

CELL_CENTRAL_LAYER_COUNT = 8
CELL_CENTRAL_LAYER_SIZE = 32

def generate_cell_context(cell_field_selections, fields_values, fields_cell_receptors, fields_receptor_ops):
    fields_receptor_outputs = []
    for values, cell_receptors, receptor_ops in zip(fields_values, fields_cell_receptors, fields_receptor_ops):
        fields_receptor_outputs.append(receptor_ops(cell_receptors, values))
    receptor_outputs = tf.reduce_sum(fields_receptor_outputs, axis=0)
    return tf.stack(cell_field_selections, receptor_outputs, axis=1)

def update_hidden(hidden_states, context_hints, cell_genes, cell_epigenetics):
    state_0 = tf.stack((hidden_states, context_hints), axis=1)
    state_1 = tf.nn.relu(tf.multiply(tf.matmul(cell_genes[0][0], state_0) + cell_genes[0][1], cell_epigenetics[0]))
    state_2 = tf.math.atan(
        tf.nn.relu(tf.multiply(tf.matmul(cell_genes[1][0], state_1) + cell_genes[1][1], cell_epigenetics[1])))
    state_3 = tf.nn.relu(tf.multiply(tf.matmul(cell_genes[2][0], state_2) + cell_genes[2][1], cell_epigenetics[2]))
    state_4 = tf.math.atan(
        tf.nn.relu(tf.multiply(tf.matmul(cell_genes[3][0], state_3) + cell_genes[3][1], cell_epigenetics[3])))
    state_5 = tf.nn.relu(tf.multiply(tf.matmul(cell_genes[4][0], state_4) + cell_genes[4][1], cell_epigenetics[4]))
    state_6 = tf.math.atan(
        tf.nn.relu(tf.multiply(tf.matmul(cell_genes[5][0], state_5) + cell_genes[5][1], cell_epigenetics[5])))
    state_7 = tf.nn.relu(tf.multiply(tf.matmul(cell_genes[6][0], state_6) + cell_genes[6][1], cell_epigenetics[6]))
    state_8 = tf.math.atan(tf.multiply(tf.matmul(cell_genes[7][0], state_7) + cell_genes[7][1], cell_epigenetics[7]))
    return state_8


def select_action(hidden_states, cell_genes, cell_action_filters):
    state_1 = tf.math.atan(tf.nn.relu(tf.matmul(cell_genes[0][0], hidden_states) + cell_genes[0][1]))
    selection_heat = tf.math.atan(tf.nn.relu(tf.matmul(cell_genes[1][0], state_1) + cell_genes[1][1]))
    masked_selection_heats = tf.multiply(selection_heat, cell_action_filters)
    return tf.argmax(masked_selection_heats, axis=1)


def perform_reset(hidden_state_list, cells):
    hidden_states = tf.stack(hidden_state_list, axis=0)
    cell_genes = (cell.provide_chromosome(ChromosomeSet.RESET) for cell in cells)
    w1 = tf.stack((reset_gene[0] for reset_gene in cell_genes))
    b1 = tf.stack((reset_gene[1] for reset_gene in cell_genes))
    w2 = tf.stack((reset_gene[2] for reset_gene in cell_genes))
    b2 = tf.stack((reset_gene[3] for reset_gene in cell_genes))
    state_1 = tf.nn.relu(tf.matmul(w1, hidden_states) + b1)
    state_2 = tf.math.atan(tf.nn.relu(tf.matmul(w2, state_1) + b2))
    for cell, field_selection in zip(cells, tf.argmax(state_2, axis=1)):
        cell.reset(field_selection)


def perform_field_shift(hidden_state_list, cells, fields):
    hidden_states = tf.stack(hidden_state_list, axis=0)
    cell_genes = (cell.provide_chromosome(ChromosomeSet.FIELD_SHIFT) for cell in cells)
    cell_field_shift_allowances = tf.stack((fields.field_shift_allowances[cell.field_index] for cell in cells))
    w1 = tf.stack((reset_gene[0] for reset_gene in cell_genes))
    b1 = tf.stack((reset_gene[1] for reset_gene in cell_genes))
    w2 = tf.stack((reset_gene[2] for reset_gene in cell_genes))
    b2 = tf.stack((reset_gene[3] for reset_gene in cell_genes))
    state_1 = tf.nn.relu(tf.matmul(w1, hidden_states) + b1)
    state_2 = tf.math.atan(tf.nn.relu(tf.matmul(w2, state_1) + b2))
    field_shift_heat_map = tf.multiply(state_2, cell_field_shift_allowances)
    for cell, field_selection in zip(cells, tf.argmax(field_shift_heat_map, axis=1)):
        cell.add_field_shift(field_selection)


def perform_projection(hidden_state_list, cells, fields):
    hidden_states = tf.stack(hidden_state_list, axis=0)
    cell_projection_allowances = tf.stack((fields.projection_allowances[cell.field_index] for cell in cells))
    cell_genes = (cell.provide_chromosome(ChromosomeSet.PROJECTION) for cell in cells)
    w1 = tf.stack((reset_gene[0] for reset_gene in cell_genes))
    b1 = tf.stack((reset_gene[1] for reset_gene in cell_genes))
    w2 = tf.stack((reset_gene[2] for reset_gene in cell_genes))
    b2 = tf.stack((reset_gene[3] for reset_gene in cell_genes))
    w3 = tf.stack((reset_gene[4] for reset_gene in cell_genes))
    b3 = tf.stack((reset_gene[5] for reset_gene in cell_genes))
    state_1 = tf.nn.relu(tf.matmul(w1, hidden_states) + b1)
    state_2 = tf.math.atan(tf.nn.relu(tf.matmul(w2, state_1) + b2))
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


def perform_einsum(hidden_state_list, cells, fields):
    hidden_states = tf.stack(hidden_state_list, axis=0)
    cell_einsum_allowances = tf.stack((fields.einsum_allowances[cell.field_index] for cell in cells))
    cell_genes = (cell.provide_chromosome(ChromosomeSet.EINSUM) for cell in cells)
    w1 = tf.stack((reset_gene[0] for reset_gene in cell_genes))
    b1 = tf.stack((reset_gene[1] for reset_gene in cell_genes))
    w2 = tf.stack((reset_gene[2] for reset_gene in cell_genes))
    b2 = tf.stack((reset_gene[3] for reset_gene in cell_genes))
    w3 = tf.stack((reset_gene[4] for reset_gene in cell_genes))
    b3 = tf.stack((reset_gene[5] for reset_gene in cell_genes))
    state_1 = tf.nn.relu(tf.matmul(w1, hidden_states) + b1)
    state_2 = tf.math.atan(tf.nn.relu(tf.matmul(w2, state_1) + b2))
    state_3 = tf.matmul(w3, state_2) + b3
    field_shift_heat_map = tf.multiply(state_3, cell_einsum_allowances)
    for cell, field_selection, key in zip(cells, tf.argmax(field_shift_heat_map, axis=1), state_2):
        cell.add_einsum(field_selection, key)


def perform_conv(hidden_state_list, cells, fields):
    hidden_states = tf.stack(hidden_state_list, axis=0)
    field_conv_selection_mask = tf.stack((fields.conv_allowances[cell.field_index] for cell in cells))
    cell_genes = (cell.provide_chromosome(ChromosomeSet.CONV) for cell in cells)
    w1 = tf.stack((reset_gene[0] for reset_gene in cell_genes))
    b1 = tf.stack((reset_gene[1] for reset_gene in cell_genes))
    w2 = tf.stack((reset_gene[2] for reset_gene in cell_genes))
    b2 = tf.stack((reset_gene[3] for reset_gene in cell_genes))
    w3 = tf.stack((reset_gene[4] for reset_gene in cell_genes))
    b3 = tf.stack((reset_gene[5] for reset_gene in cell_genes))
    state_1 = tf.nn.relu(tf.matmul(w1, hidden_states) + b1)
    state_2 = tf.math.atan(tf.nn.relu(tf.matmul(w2, state_1) + b2))
    state_3 = tf.matmul(w3, state_2) + b3
    conv_heat_map = tf.multiply(state_3, field_conv_selection_mask)
    for cell, field_selection, key in zip(cells, tf.argmax(conv_heat_map, axis=1), state_2):
        cell.add_conv(field_selection, key)


def perform_move(hidden_state_list, cells):
    hidden_states = tf.stack(hidden_state_list, axis=0)
    cell_genes = (cell.provide_chromosome(ChromosomeSet.MOVE) for cell in cells)
    w1 = tf.stack((reset_gene[0] for reset_gene in cell_genes))
    b1 = tf.stack((reset_gene[1] for reset_gene in cell_genes))
    w2 = tf.stack((reset_gene[2] for reset_gene in cell_genes))
    b2 = tf.stack((reset_gene[3] for reset_gene in cell_genes))
    state_1 = tf.nn.relu(tf.matmul(w1, hidden_states) + b1)
    state_2 = tf.matmul(w2, state_1) + b2
    for cell, direction in zip(cells, state_2):
        cell.move(direction[0], direction[1])


def perform_mate(hidden_state_list, cells, mating_list):
    hidden_states = tf.stack(hidden_state_list, axis=0)
    cell_genes = (cell.provide_chromosome(ChromosomeSet.MATE) for cell in cells)
    w1 = tf.stack((reset_gene[0] for reset_gene in cell_genes))
    b1 = tf.stack((reset_gene[1] for reset_gene in cell_genes))
    w2 = tf.stack((reset_gene[2] for reset_gene in cell_genes))
    b2 = tf.stack((reset_gene[3] for reset_gene in cell_genes))
    state_1 = tf.nn.relu(tf.matmul(w1, hidden_states) + b1)
    state_2 = tf.matmul(w2, state_1) + b2
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
    state_1 = tf.nn.relu(tf.matmul(w1, hidden_states) + b1)
    state_2 = tf.matmul(w2, state_1) + b2
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
    state_1 = tf.nn.relu(tf.matmul(w1, hidden_states) + b1)
    state_2 = tf.matmul(w2, state_1) + b2
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
    state_1 = tf.nn.relu(tf.matmul(w1, hidden_states) + b1)
    state_2 = tf.matmul(w2, state_1) + b2
    cells_direction, cells_key = tf.split(state_2, [1, 4])
    for cell, direction, key in zip(cells, cells_direction, cells_key):
        if cell.mate():
            transfer_list.append((cell, direction, key))

def operate(cells, fields, cell_receptors, cell_main_genes, cell_hidden_states, cell_selector_genes, mating_list, transfer_list):
    field_selections = tf.TensorArray(dtype=tf.float32, size=len(cells))
    cell_action_filters = tf.TensorArray(dtype=tf.float32, size=len(cells))
    for cell_index, cell in enumerate(cells):
        field_selection = np.zeros(len(fields))
        field_selection[cell.field_index] = 1
        field_selections = field_selections.write(cell_index, field_selection)
        cell_action_filters = cell_action_filters.write(cell_index, cell.action_mask)
    context_hints = generate_cell_context(
        field_selections.stack(),
        [field.value for field in fields],
        cell_receptors,
        [field.receptor_ops for field in fields]
    )
    cell_epigenetics = []
    for epigene_index in range(CELL_CENTRAL_LAYER_COUNT):
        epigene = tf.TensorArray(dtype=tf.float32, size=len(cells))
        for cell_index, cell in enumerate(cells):
            epigene = epigene.write(cell_index, cell.provide_epigenes(epigene_index))
        cell_epigenetics.append(epigene.stack())
    new_hidden = update_hidden(cell_hidden_states, context_hints, cell_main_genes, cell_epigenetics)
    cell_selections = select_action(new_hidden, cell_selector_genes, cell_action_filters)
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
        wait_hole
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
        (data[0] for data in projection_data),
        fields
    )
    for cell_lock_data in softmax_data:
        cell_lock_data[0].add_softmax()
    perform_einsum(
        tf.stack((data[1] for data in einsum_data)),
        (data[0] for data in einsum_data),
        fields
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
    """
        add_epigene_data,
        remove_epigene_data,
        lock_data,
        unlock_data,
        transfer_energy_data,
        wait_hole
    """
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
