import numpy as np

def prepare_batch(inputs):
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    max_sequence_length = max(sequence_lengths)
        
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], 
                                  dtype=np.int32)

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    return inputs_batch_major, sequence_lengths

        
def batch_generator(x, y):            
    while True:
        i = np.random.randint(0, len(x))
        yield [x[i], y[i]]
        
def input_generator(x, y, batch_size):
    gen_batch = batch_generator(x, y)

    x_batch = []
    y_batch = []
    for i in range(batch_size):
        a, b= next(gen_batch)
        x_batch += [a]
        y_batch += [b]
    return x_batch, y_batch