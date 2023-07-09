from enum import IntEnum


class ChromosomeSet(IntEnum):
    MAIN = 0
    SELECT = 1
    EMIT = 2
    ACTION_OPERATOR_SET = 3
    SINGLE_FIELD_SELECTOR = 4
    SINGLE_FIELD_SELECTOR_AND_KEY = 5
    DOUBLE_FILED_SELECTOR = 6
    DIRECTION_SELECTOR = 7
    DIRECTION_SELECTOR_AND_KEY = 8
    RECEIVE_ENERGY = 9
    SELECT_EPIGENE = 10
    RECEPTORS = 11


class ActionSet(IntEnum):
    RESET = 0
    FIELD_SHIFT = 1
    PROJECTION = 2
    SOFTMAX = 3
    EINSUM = 4
    CONV = 5
    MOVE = 6
    MATE = 7
    ADD_EPIGENE = 8
    REMOVE_EPIGENE = 9
    LOCK = 10
    UNLOCK = 11
    TRANSFER_ENERGY = 12
    WAIT = 13
    CONCAT = 14
    MULTIPLY = 15
    ADD = 16
    BELL = 17