import tensorflow as tf
from Field import Field


def make_ecosystem(fields_shapes):
    return Ecosystem(fields_shapes)


class Ecosystem:
    def __init__(self, fields_shapes):
        self._fields = tuple(Field(fields_shape, field_index) for field_index, fields_shape in enumerate(fields_shapes))
        for field in self._fields:
            field.build_graphs(self._fields)
