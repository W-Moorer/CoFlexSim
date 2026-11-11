"""Pure slot planning: fill holes before replacing inactive response columns."""
import numpy as np


def plan_slots(slots, requested):
    slots = np.asarray(slots, dtype=np.int64)
    requested = np.unique(np.asarray(requested, dtype=np.int64))
    if np.any(requested < 0) or len(requested) > len(slots):
        raise ValueError('Invalid or oversized response working region')
    present = slots[slots >= 0]
    if len(np.unique(present)) != len(present):
        raise ValueError('Duplicate resident response columns')
    missing = requested[~np.isin(requested, present)]
    empty = np.flatnonzero(slots < 0)
    inactive = np.flatnonzero((slots >= 0) & ~np.isin(slots, requested))
    destinations = np.concatenate((empty, inactive))[:len(missing)]
    if len(destinations) != len(missing):
        raise RuntimeError('Insufficient evictable response slots')
    return missing, destinations, slots[destinations][slots[destinations] >= 0]
