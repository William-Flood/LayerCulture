import numpy as np
import tensorflow as tf

class Field:
    def __init__(self, shape, field_index):
        self._shape = shape
        self._field_index = field_index
        self._reshape_list = None
        self._project_list = None
        self._einsum_list = None
        self._action_mask = None
        self._dimensional_shift = None
        self._add_projection = None
        self._einsum = None
        self._make_conv = None
        self._make_concat = None
        self._make_divide = None
        self._make_add = None

        self._field_shift_allowances = None
        self._projection_allowances = None
        self._einsum_allowances = None
        self._conv_allowances = None
        self._concat_allowances = None
        self._elementwise_allowances = None
        self._einsum_allowances_with = dict()
        self.einsum_string_dict = dict()

    @property
    def field_index(self):
        return self._field_index

    @property
    def shape(self):
        return self._shape

    @property
    def reshape_list(self):
        return self._reshape_list

    @reshape_list.setter
    def reshape_list(self, value):
        self._reshape_list = value

    @property
    def project_list(self):
        return self._project_list

    @project_list.setter
    def project_list(self, value):
        self._project_list = value

    def get_einsum_list(self):
        return self._einsum_list

    @property
    def action_mask(self):
        return self._action_mask

    @property
    def dimensional_shift(self):
        return self._dimensional_shift

    @property
    def add_projection(self):
        return self._add_projection

    @property
    def einsum(self):
        return self._einsum

    @property
    def make_conv(self):
        return self._make_conv

    @make_conv.setter
    def make_conv(self, value):
        self._make_conv = value

    @property
    def make_concat(self):
        return self._make_concat

    @make_concat.setter
    def make_concat(self, value):
        self._make_concat = value

    @property
    def make_divide(self):
        return self._make_divide

    @make_divide.setter
    def make_divide(self, value):
        self._make_divide = value

    @property
    def make_add(self):
        return self._make_add

    @make_add.setter
    def make_add(self, value):
        self._make_add = value

    def softmax(self, cell_op_state):
        cell_op_state[self._field_index] = tf.nn.softmax(cell_op_state[self._field_index], axis=-1)

    @property
    def field_shift_allowances(self):
        return self._field_shift_allowances

    @property
    def einsum_allowances_with(self):
        return self._einsum_allowances_with

    def build_field_shift_ops(self, fields):
        this_shape_prod = int(np.prod(self._shape))
        fields_compatible = tuple(
            int(np.prod(field.shape)) == this_shape_prod and field.field_index != self._field_index for field in fields
        )
        self._dimensional_shift = (
            self.build_field_shift_op(field) if fields_compatible[field.field_index] else None for field in fields
        )
        self._field_shift_allowances = (1.0 if field_compatible else 0.0 for field_compatible in fields_compatible)

    def build_field_shift_op(self, field):
        field_index = self._field_index
        other_index = field.field_index
        def field_shift_op(cell_op_state):
            cell_op_state[other_index] = tf.reshape(cell_op_state[field_index], [-1] + field.shape)
        return field_shift_op

    def build_projection_ops(self, fields):
        fields_compatible = tuple(
            len(field.shape) == len(self._shape) and field.shape[-1] == self._shape[-1] and field.field_index != self._field_index for field in fields
        )
        self._add_projection = (
            self.build_projection_op(field) if fields_compatible[field.field_index] else None for field in fields
        )
        self._projection_allowances = (1.0 if field_compatible else 0.0 for field_compatible in fields_compatible)

    def build_projection_op(self, field):
        field_index = self._field_index
        other_index = field.field_index
        w_1 = tf.random.normal([8, 32], seed=65)
        b_1 = tf.random.normal([32], seed=65)
        w_2 = tf.random.normal([32, field.shape[-1]], seed=65)
        b_2 = tf.random.normal([field.shape[-1]], seed=65)
        def projection_op_generator(projection_key):
            indices = tf.matmul(tf.nn.relu(tf.matmul(projection_key, w_1) + b_1), w_2) + b_2
            def projection_op(cell_op_state):
                cell_op_state[other_index] = tf.gather(cell_op_state[field_index], indices, axis=-1)
            return projection_op
        return projection_op_generator

    def build_conv_ops(self, fields):
        fields_compatible = tuple(
            len(field.shape) == len(self._shape) and field.field_index != self._field_index for field in fields
        )

        self._make_conv = tuple(self.build_conv_op(field) if these_fields_compatible else None
                                for these_fields_compatible, field in zip(fields_compatible, fields))
        self._conv_allowances = (1.0 if field_compatible else 0.0 for field_compatible in fields_compatible)

    def generate_conv_params(self, other_field):
        # define parameters for convolution
        conv_filter_shape = [3, 3, self._shape[-1], other_field.shape[-1]]
        seed = 42

        # generate random filter weights and biases
        filter_weights = tf.random.normal(conv_filter_shape, seed=seed)
        filter_biases = tf.random.normal([other_field.shape[-1]], seed=seed)

        # return parameters as a tuple
        return filter_weights, filter_biases, other_field.field_index

    def build_conv_op(self, other_field):
        filter_weights, filter_biases, other_index = self.generate_conv_params(other_field)

        def conv_op(cell_op_state):
            conv_output = tf.nn.conv2d(cell_op_state[self._field_index], filter_weights, strides=[1, 1, 1, 1],
                                       padding='SAME')
            conv_output = tf.nn.bias_add(conv_output, filter_biases)
            cell_op_state[other_index] = conv_output

        return conv_op

    def field_compatible_for_einsum(self, other, ecosystem_fields):
        self_dims = len(self._shape)
        other_dims = len(other.shape)
        common_dims = min(self_dims, other_dims)
        self_axes = "".join(chr(98 + j) for j in range(self_dims))
        dims_to = common_dims if self_dims == other_dims else common_dims + 1

        full_match = False
        compatible_fields = [0.0] * len(ecosystem_fields)
        op_strings = [""] * len(ecosystem_fields)
        for i in range(dims_to):
            self_fields_to_match = self._shape[-1*i:]
            other_fields_to_match = other.shape[:i]
            other_axes = "".join(chr(98 + j + self_dims - i) for j in range(other_dims))
            einsum_string = f"a{self_axes},a{other_axes}->a{self_axes[:self_dims-i]}{other_axes[i:]}"

            dim_match = True
            for j in zip(self_fields_to_match, other_fields_to_match):
                dim_match = dim_match and j[0] == j[1]

            if dim_match:
                output_shape = self._shape[:self_dims-i] + other.shape[i:]
                for ecosystem_field_index, ecosystem_field in enumerate(ecosystem_fields):
                    if len(output_shape) == len(ecosystem_field.shape):
                        dim_matches = [output_dim_size == field_dim_size
                                                 for output_dim_size, field_dim_size in zip(output_shape, ecosystem_field.shape)]
                        to_with_compatible = all(dim_matches)

                        if to_with_compatible:
                            full_match = True
                            compatible_fields[ecosystem_field_index] = 1.0
                            op_strings[ecosystem_field_index] = einsum_string
                            break

        return full_match, compatible_fields, op_strings, other

    def build_einsum_op(self, other_with, other_to):
        with_index = other_with.field_index
        to_index = other_to.field_index
        this_index = self._field_index
        einsum_string = self.einsum_string_dict[with_index][to_index]
        def einsum_op(cell_op_state):
            cell_op_state[to_index] = tf.einsum(einsum_string, cell_op_state[this_index], cell_op_state[with_index])
        return einsum_op

    def build_einsum_ops(self, ecosystem_fields):
        einsum_compatability_props = tuple(self.field_compatible_for_einsum(other_field, ecosystem_fields)
                                           for other_field in ecosystem_fields)

        for field_index, field_einsum_compatability_props in enumerate(einsum_compatability_props):
            if field_einsum_compatability_props[0]:
                self.einsum_string_dict[field_index] = field_einsum_compatability_props[2]
                self.einsum_allowances_with[field_index] = field_einsum_compatability_props[1]

    @property
    def projection_allowances(self):
        return self._projection_allowances

    @property
    def einsum_allowances(self):
        return self._einsum_allowances

    @property
    def conv_allowances(self):
        return self._conv_allowances

    @property
    def concat_allowances(self):
        return self._concat_allowances

    @property
    def elementwise_allowances(self):
        return self._elementwise_allowances

    def build_elementwise_ops(self, fields):
        elementwise_compatible = \
            [len(self._shape) == len(other.shape) and all(
                this_dim_size == other_dim_size for this_dim_size, other_dim_size in zip(self._shape, other.shape)
            ) for other in fields]

        self._elementwise_allowances = [1.0 if compatible else 0 for compatible in elementwise_compatible]
        self._make_add = [self.build_add_op(other) if compatible else None
                         for other, compatible in zip(fields, elementwise_compatible)]
        self._make_divide = [self.build_divide_op(other) if compatible else None
                         for other, compatible in zip(fields, elementwise_compatible)]

    def build_add_op(self, other):
        other_index = other.field_index
        this_index = self._field_index
        def add_op(cell_op_state):
            cell_op_state[this_index] = cell_op_state[this_index] = cell_op_state[other_index]
        return add_op

    def build_divide_op(self, other):
        other_index = other.field_index
        this_index = self._field_index
        def divide_op(cell_op_state):
            cell_op_state[this_index] = tf.divide(cell_op_state[this_index], cell_op_state[other_index])

        return divide_op

    def build_graphs(self, fields):
        self.build_field_shift_ops(fields)
        self.build_projection_ops(fields)
        self.build_einsum_ops(fields)
        self.build_elementwise_ops(fields)
