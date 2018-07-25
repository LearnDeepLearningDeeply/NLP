import numpy as np

PAD_ID = 0
START_ID = 1
np.random.seed(2358)

class DataIterator:
    def __init__(self, model, data_set, batch_size):
        self.data_set = data_set
        self.batch_size = batch_size
        self.model = model

    def next_random(self):
        # first random bucket, then random sentences
        while True:
            source_inputs, target_outputs, source_lengths, _ = self.model.get_batch(self.data_set)
            yield source_inputs, source_lengths, target_outputs

    def next_sequence(self, test = False):
        # select sentence one by one
        start_id = 0
        while True:

            if test:
                get_batch_func = self.model.get_test_batch

                source_inputs, source_lengths, finished = self.model.get_test_batch(self.data_set, start_id = start_id)
        
                yield source_inputs, source_lengths

            else:
                get_batch_func = self.model.get_batch
                source_inputs, target_outputs, source_lengths, finished = self.model.get_batch(self.data_set, start_id = start_id)

                yield source_inputs, source_lengths, target_outputs

            if finished:
                break
            start_id += self.batch_size

