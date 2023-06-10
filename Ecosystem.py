import tensorflow as tf
from Field import Field
from GenerateRandomChromosomes import create_random_genome
from Cell import Cell
import random
from CellBatchOperator import CELL_CENTRAL_LAYER_SIZE, CELL_CENTRAL_LAYER_COUNT, operate


def make_ecosystem(fields_shapes):
    return Ecosystem(fields_shapes)


class Ecosystem:
    def __init__(self, fields_shapes, initial_cell_groups=1000, generated_family_size=10):
        self._fields = tuple(Field(fields_shape, field_index) for field_index, fields_shape in enumerate(fields_shapes))
        self.cells = []
        for field in self._fields:
            field.build_graphs(self._fields)
        self.cell_positions = []

        for group_index in range(initial_cell_groups):
            genome = create_random_genome(len(self._fields), hidden_size=CELL_CENTRAL_LAYER_SIZE)
            group_w = random.random()
            group_x = random.random()
            group_y = random.random()
            group_z = random.random()
            if 0 == group_index % 100:
                print(f"Populating group {group_index}")

            for _ in range(generated_family_size):
                cw = group_w + .01 * random.random()
                cx = group_x + .01 * random.random()
                cy = group_y + .01 * random.random()
                cz = group_z + .01 * random.random()
                self.cell_positions.append([cw, cx, cy, cz])
                self.cells.append(
                    Cell(self._fields, genome, genome, .1,
                         tf.Variable(tf.ones([CELL_CENTRAL_LAYER_COUNT, CELL_CENTRAL_LAYER_SIZE], dtype=tf.float32)),
                         cw,
                         cx,
                         cy,
                         cz)
                )

    def simulate(self, simulation_steps):
        hidden_states = tf.zeros([len(self.cells), CELL_CENTRAL_LAYER_SIZE])
        cell_signals = tf.zeros([len(self.cells), 2, 8])
        cell_positions_tensor = tf.stack(self.cell_positions)
        for _ in range(simulation_steps):
            hidden_states, cell_positions_tensor, cell_signals = operate(self.cells, self._fields, hidden_states, cell_signals, [], [], cell_positions_tensor)

