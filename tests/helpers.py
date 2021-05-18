import numpy


def nested_arrays_equal(a, b):
    """
    Function to compare two (nested) lists/tuples of numpy arrays a and b
    from: https://stackoverflow.com/questions/31662528/compare-two-nested-lists-tuples-of-numpy-arrays/31662874#31662874
    """

    if isinstance(a, (list, tuple)) and isinstance(b, type(a)):
        for aa, bb in zip(a, b):
            if not nested_arrays_equal(aa, bb):
                return False  # Short circuit
        return True
    else:  # numpy arrays
        return numpy.all(a == b)
