from CellConstants import ActionSet
import tensorflow as tf

def make_executor(actionType : ActionSet, left_field, right_field, return_field, generation_state):
    match actionType:
        case ActionSet.FIELD_SHIFT:
            op = create_field_shift(return_field, left_field)
            param_count = 1
        case ActionSet.PROJECTION:
            op = create_projection(return_field, generation_state, left_field)
            param_count = 1
        case ActionSet.SOFTMAX:
            op = create_softmax(left_field)
            param_count = 1
        case ActionSet.EINSUM:
            op = create_einsum(right_field, return_field, left_field)
            param_count = 2
        case ActionSet.CONCAT:
            op = create_concat(right_field, return_field, left_field)
            param_count = 2
        case ActionSet.MULTIPLY:
            op = create_multiply(right_field, left_field)
            param_count = 2
        case ActionSet.ADD:
            op = create_add(right_field, left_field)
            param_count = 2
        case ActionSet.BELL:
            op = create_bell(left_field)
            param_count = 1
        case ActionSet.CONV:
            op = create_conv(return_field, generation_state, left_field)
            param_count = 1
        case _:
            raise "Unrecognized action type provided"

    if 1 == param_count:
        # @tf.function
        def ops(cell_op_state):
            op_result = op(cell_op_state[left_field.field_index])
            clipped_result = tf.clip_by_value(
                op_result,
                clip_value_min=-1e4,
                clip_value_max=1e4
            )
            return clipped_result
        return ops
    elif 2 == param_count:
        # @tf.function
        def ops(cell_op_state):
            op_result = op(cell_op_state[left_field.field_index], cell_op_state[right_field.field_index])
            clipped_result = tf.clip_by_value(
                op_result,
                clip_value_min=-1e4,
                clip_value_max=1e4
            )
            return clipped_result
        return ops

def create_concat(right_field, return_field, left_field):
    return left_field.make_concat(right_field, return_field)


def create_multiply(right_field, left_field):
    return left_field.make_multiply[right_field.field_index]

def create_add(right_field, left_field):
    return left_field.make_add[right_field.field_index]
    

def create_field_shift(return_field, left_field):
    return left_field.dimensional_shift[return_field.field_index]

def create_projection(return_field, projection_key, left_field):
    return left_field.add_projection[return_field.field_index](projection_key)

def create_softmax(left_field):
    return left_field.softmax

def create_bell(left_field):
    return left_field.bell

def create_einsum(right_field, return_field, left_field):
    return left_field.build_einsum_op(right_field, return_field)

def create_conv(return_field, conv_gen_state, left_field):
    return left_field.make_conv[return_field.field_index](conv_gen_state)
