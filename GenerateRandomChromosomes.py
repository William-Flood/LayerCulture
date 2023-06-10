import tensorflow as tf
from CellBatchOperator import GENERATION_KEY_SIZE


def create_random_genome(field_count, hidden_size, signal_size=8):
    return [
        create_main_chromosome(field_count, hidden_size, signal_size),
        create_reset_chromosome(field_count, hidden_size),
        create_field_shift_chromosome(field_count, hidden_size),
        create_projection_chromosome(field_count, hidden_size),
        create_einsum_chromosome(field_count, hidden_size),
        create_conv_chromosome(field_count, hidden_size),
        create_move_chromosome(hidden_size),
        create_mate_chromosome(hidden_size),
        create_add_epigene_chromosome(hidden_size),
        create_remove_epigene_chromosome(hidden_size),
        create_transfer_energy_chromosome(hidden_size),
        create_receive_energy_chromosome(hidden_size),
        create_receptors_chromosome(signal_size),
        create_concat_chromosome(field_count, hidden_size),
        create_select_chromosome(hidden_size),
        create_divide_chromosome(field_count, hidden_size),
        create_add_chromosome(field_count, hidden_size),
        create_emit_chromosome(hidden_size, signal_size)
    ]


def create_main_chromosome(field_count, hidden_size, signal_size):
    CELL_SCALAR_CONTEXTS = 4
    return [
        tf.random.normal([hidden_size + field_count + signal_size + CELL_SCALAR_CONTEXTS, hidden_size]),
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


def create_reset_chromosome(field_count, hidden_size):
    return [
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, field_count]),
        tf.random.normal([field_count]),
    ]


def create_field_shift_chromosome(field_count, hidden_size):
    return [
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, field_count]),
        tf.random.normal([field_count]),
    ]


def create_projection_chromosome(field_count, hidden_size):
    return [
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, field_count + GENERATION_KEY_SIZE]),
        tf.random.normal([field_count + GENERATION_KEY_SIZE]),
    ]


def create_einsum_chromosome(field_count, hidden_size):
    return [
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, field_count * 2]),
        tf.random.normal([field_count * 2]),
    ]


def create_conv_chromosome(field_count, hidden_size):
    return [
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        # Includes the kernel and bias key
        tf.random.normal([hidden_size, field_count + GENERATION_KEY_SIZE * 2]),
        tf.random.normal([field_count + GENERATION_KEY_SIZE * 2])
    ]


def create_move_chromosome(hidden_size):
    return [
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, 4]),
        tf.random.normal([4]),
    ]


def create_mate_chromosome(hidden_size):
    return [
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, 4]),
        tf.random.normal([4]),
    ]


def create_add_epigene_chromosome(hidden_size):
    return [
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, 40]),
        tf.random.normal([40]),
    ]


def create_remove_epigene_chromosome(hidden_size):
    return [
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, 40]),
        tf.random.normal([40]),
    ]


def create_transfer_energy_chromosome(hidden_size):
    return [
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, 8]),
        tf.random.normal([8]),
    ]


def create_receive_energy_chromosome(hidden_size):
    return [
        tf.random.normal([36, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, 2]),
        tf.random.normal([2]),
    ]


def create_receptors_chromosome(signal_size):
    return [
        tf.random.normal([2, signal_size, signal_size]),
        tf.random.normal([signal_size]),
        tf.random.normal([signal_size, signal_size]),
        tf.random.normal([signal_size]),
    ]


def create_concat_chromosome(field_count, hidden_size):
    return [
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, field_count * 2]),
        tf.random.normal([field_count * 2]),
    ]


def create_select_chromosome(hidden_size):
    ACTION_COUNT = 17
    return [
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, ACTION_COUNT]),
        tf.random.normal([ACTION_COUNT]),
    ]


def create_divide_chromosome(field_count, hidden_size):
    return [
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, field_count]),
        tf.random.normal([field_count]),
    ]


def create_add_chromosome(field_count, hidden_size):
    return [
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, field_count]),
        tf.random.normal([field_count]),
    ]


def create_emit_chromosome(hidden_size, signal_size):
    return [
        tf.random.normal([hidden_size, hidden_size]),
        tf.random.normal([hidden_size]),
        tf.random.normal([hidden_size, signal_size]),
        tf.random.normal([signal_size]),
    ]
