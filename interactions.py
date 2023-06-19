import numpy as np
import scipy.sparse as sp


class Interactions(object):


    def __init__(self, user_item_sequence, num_users, num_items):
        user_ids, item_ids = [], []
        for uid, item_seq in enumerate(user_item_sequence):
            for iid in item_seq:
                user_ids.append(uid)
                item_ids.append(iid)

        user_ids = np.asarray(user_ids)
        item_ids = np.asarray(item_ids)

        self.num_users = num_users
        self.num_items = num_items

        self.user_ids = user_ids
        self.item_ids = item_ids

        self.sequences = None
        self.test_sequences = None

    def __len__(self):
        return len(self.user_ids)

    def tocoo(self):
        row = self.user_ids
        col = self.item_ids
        data = np.ones(len(self))

        return sp.coo_matrix((data, (row, col)),
                             shape=(self.num_users, self.num_items))

    def tocsr(self):
         return self.tocoo().tocsr()

    def to_sequence(self, sequence_length=5, target_length=1):
        max_sequence_length = sequence_length + target_length

        # Sort first by user id
        sort_indices = np.lexsort((self.user_ids,))

        user_ids = self.user_ids[sort_indices]
        item_ids = self.item_ids[sort_indices]

        user_ids, indices, counts = np.unique(user_ids,
                                              return_index=True,
                                              return_counts=True)

        num_subsequences = sum([c - max_sequence_length + 1 if c >= max_sequence_length else 1 for c in counts])

        # max_number = max(counts)-max_sequence_length
        max_number = 20
        left_sequence = np.zeros((num_subsequences, max_number),
                             dtype=np.int64)

        sequences = np.zeros((num_subsequences, sequence_length),
                             dtype=np.int64)
        sequences_targets = np.zeros((num_subsequences, target_length),
                                     dtype=np.int64)
        sequence_users = np.empty(num_subsequences,
                                  dtype=np.int64)

        test_sequences = np.zeros((self.num_users, sequence_length),
                                  dtype=np.int64)
        test_left_sequence = np.zeros((self.num_users, max_number+target_length),
                             dtype=np.int64)
        test_users = np.empty(self.num_users,
                              dtype=np.int64)

        _uid = None
        for i, (uid,
                item_seq,item_seq2) in enumerate(_generate_sequences(user_ids,
                                                           item_ids,
                                                           indices,
                                                           max_sequence_length,max_number)):
            if uid != _uid:
                test_sequences[uid][:] = item_seq[-sequence_length:]
                right_sequence = item_seq[:max_sequence_length-sequence_length]
                test_left_sequence[uid][:] = np.concatenate((item_seq2, right_sequence), axis=0)
                test_users[uid] = uid
                _uid = uid
            sequences_targets[i][:] = item_seq[-target_length:]
            sequences[i][:] = item_seq[:sequence_length]
            sequence_users[i] = uid
            left_sequence[i][:] = item_seq2[:]

        self.sequences = SequenceInteractions(sequence_users, sequences, left_sequence, sequences_targets)
        self.test_sequences = SequenceInteractions(test_users, test_sequences,test_left_sequence)


class SequenceInteractions(object):
    def __init__(self,
                 user_ids,
                 sequences,left_sequence,
                 targets=None):
        self.user_ids = user_ids
        self.sequences = sequences
        self.targets = targets
        self.left_sequence = left_sequence

        self.L = sequences.shape[1]
        self.T = None
        if np.any(targets):
            self.T = targets.shape[1]


def _sliding_window(tensor, window_size, max_number, step_size=1):
    if len(tensor) - window_size >= 0:
        for i in range(len(tensor), 0, -step_size):
            if i - window_size >= 0:
                left = tensor[:i-window_size]
                if len(left) <= max_number:
                    left_now = np.pad(left,(max_number-len(left), 0), 'constant')
                else:
                    left_now = left[-max_number:]
                # left_now = np.pad(left,(max_number-len(left),0),'constant')
                yield (tensor[i - window_size:i], left_now)
            else:
                break
    else:
        num_paddings = window_size - len(tensor)
        # Pad sequence with 0s if it is shorter than windows size.
        yield (np.pad(tensor, (num_paddings, 0), 'constant'),np.pad(tensor, (max_number-len(tensor), 0), 'constant'))


def _generate_sequences(user_ids, item_ids,
                        indices,
                        max_sequence_length, max_number):
    for i in range(len(indices)):

        start_idx = indices[i]
        # start_idx = indices[0]
        #
        if i >= len(indices) - 1:
            stop_idx = None
        else:
            stop_idx = indices[i + 1]

        for (seq, seq2) in _sliding_window(item_ids[start_idx:stop_idx],
                                   max_sequence_length,max_number):
            yield (user_ids[i], seq, seq2)
