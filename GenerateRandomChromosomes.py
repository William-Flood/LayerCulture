import tensorflow as tf
from CellConstants import ActionSet
from CellBatchOperator import CELL_CENTRAL_LAYER_COUNT, CELL_CENTRAL_LAYER_SIZE
from Cell import OP_GEN_STATE_SIZE




def create_random_genome(field_count, hidden_size, signal_size=8):
    return [
        create_main_chromosome(field_count, hidden_size, signal_size, OP_GEN_STATE_SIZE),
        create_select_chromosome(hidden_size),
        create_emit_chromosome(hidden_size, signal_size),
        create_action_operator_set_chromosomes(hidden_size, len(ActionSet)),
        create_single_field_selector_chromosomes(hidden_size, field_count),
        create_double_field_selector_chromosomes(hidden_size, field_count),
        create_double_field_selector_and_key_chromosomes(hidden_size, field_count, OP_GEN_STATE_SIZE),
        create_triple_field_selector_chromosomes(hidden_size, field_count),
        create_direction_selector_chromosomes(hidden_size),
        create_direction_selector_and_key_chromosomes(hidden_size, OP_GEN_STATE_SIZE),
        create_receive_energy_chromosome(hidden_size, OP_GEN_STATE_SIZE),
        create_select_epigene_chromosome(hidden_size),
        create_receptors_chromosome(signal_size)
    ]


def create_main_chromosome(field_count, hidden_size, signal_size, key_size):
    CELL_SCALAR_CONTEXTS = 4
    ENVIRONMENT_DIMENSIONS = 4
    main_chromosome_input_size = hidden_size + signal_size + field_count * 3 + len(ActionSet) + ENVIRONMENT_DIMENSIONS + CELL_SCALAR_CONTEXTS + key_size
    return [
        tf.random.normal([main_chromosome_input_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size])
    ]


def create_select_chromosome(hidden_size):
    return [
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, len(ActionSet)]),
        tf.random.normal([len(ActionSet)]),
    ]

def create_emit_chromosome(hidden_size, signal_size):
    return [
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, signal_size]),
        tf.random.normal([signal_size]),
    ]


def create_action_operator_set_chromosomes(hidden_size, action_count):
    return [
        tf.random.normal([action_count, hidden_size, hidden_size]),
        tf.random.normal([action_count, hidden_size]),
        tf.random.normal([action_count, hidden_size, hidden_size]),
        tf.random.normal([action_count, hidden_size])
    ]


def create_single_field_selector_chromosomes(hidden_size, field_length):
    return [
        tf.random.normal([hidden_size, field_length]),
        tf.random.normal([field_length])
    ]

def create_double_field_selector_chromosomes(hidden_size, field_length):
    return [
        tf.random.normal([hidden_size, field_length * 2]),
        tf.random.normal([field_length * 2])
    ]


def create_double_field_selector_and_key_chromosomes(hidden_size, field_length, key_size):
    return [
        tf.random.normal([hidden_size, field_length * 2 + key_size]),
        tf.random.normal([field_length * 2 + key_size])
    ]


def create_triple_field_selector_chromosomes(hidden_size, field_length):
    return [
        tf.random.normal([hidden_size, field_length * 3]),
        tf.random.normal([field_length * 3])
    ]


def create_direction_selector_chromosomes(hidden_size, environment_dims=4):
    return [
        tf.random.normal([hidden_size, environment_dims]),
        tf.random.normal([environment_dims])
    ]


def create_direction_selector_and_key_chromosomes(hidden_size, key_size, environment_dims=4):
    return [
        tf.random.normal([hidden_size, environment_dims + key_size]),
        tf.random.normal([environment_dims + key_size])
    ]


def create_receive_energy_chromosome(hidden_size, key_size):
    return [
        tf.random.normal([hidden_size + key_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, 2]),
        tf.random.normal([2]),
    ]

def create_select_epigene_chromosome(hidden_size):
    return [
        tf.random.normal([hidden_size, CELL_CENTRAL_LAYER_COUNT + CELL_CENTRAL_LAYER_SIZE]),
        tf.random.normal([CELL_CENTRAL_LAYER_COUNT + CELL_CENTRAL_LAYER_SIZE]),
    ]

def create_receptors_chromosome(signal_size):
    return [
        tf.random.normal([2, signal_size, signal_size]),
        tf.random.normal([signal_size]),
        tf.random.normal([signal_size, signal_size]),
        tf.random.normal([signal_size]),
    ]