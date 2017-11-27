import numpy as np

def prepare_batch(inputs, dim, max_sequence_length=None, emb_size = 0, time_major = False):
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
   
    if dim == 3:
        inputs_batch_major = np.zeros(shape=[batch_size, 
                                             max_sequence_length, 
                                             emb_size], 
                                      dtype=np.float32)
    elif dim == 2:
        inputs_batch_major = np.zeros(shape=[batch_size, 
                                             max_sequence_length], 
                                      dtype=np.int32)
    
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    if time_major == True:
        return inputs_batch_major.swapaxes(0, 1), sequence_lengths
    else:
        return inputs_batch_major, sequence_lengths

        
def batch_generator(seq, seq1, seq2, seq3):            
    while True:
        i = np.random.randint(0, len(seq))
        yield [seq[i], seq1[i], seq2[i], seq3[i]]
        
def input_generator(seq, seq1, seq2, seq3, batch_size):
    gen_batch = batch_generator(seq, seq1, seq2, seq3)
    
    r = []
    r1 = []
    r2 = []
    r3 = []

    for i in range(batch_size):
        a, b, c, d = next(gen_batch)
        r += [a]
        r1 += [b]
        r2 += [c]
        r3 += [d]
    return r, r1, r2, r3