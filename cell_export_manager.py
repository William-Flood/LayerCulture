import numpy as np
from CellConstants import ActionSet, OP_GEN_STATE_SIZE, ACTION_LEFT_RIGHT_RETURN, ENVIRONMENT_DIMENSIONS

def cell_data_to_array(action_type,
                                  left_index,
                                  right_index,
                                  return_index,
                                  generation_state,
                                  position):
    return np.concatenate([
        np.array([
            action_type,
            left_index,
            right_index,
            return_index
        ], dtype=np.float32),
        generation_state,
        position
    ])

def array_to_cell_data(exported_array):
    action_index = int(exported_array[0])
    if -1 == action_index:
        action_type = None
    else:
        action_type = ActionSet(action_index)
    left_index = int(exported_array[1])
    right_index = int(exported_array[2])
    return_index = int(exported_array[3])
    generation_state = exported_array[ACTION_LEFT_RIGHT_RETURN:ACTION_LEFT_RIGHT_RETURN+OP_GEN_STATE_SIZE]
    position = exported_array[-1 * ENVIRONMENT_DIMENSIONS:]
    return action_type, left_index, right_index, return_index, generation_state, position

def start_data():
    return np.zeros([ACTION_LEFT_RIGHT_RETURN + OP_GEN_STATE_SIZE + ENVIRONMENT_DIMENSIONS])