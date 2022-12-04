
import tensorflow as tf

AUTO = tf.data.AUTOTUNE

class Bert4RecDataLoader:
    """
    Class that iterates over tfrecords in order to get the sequences.
    """
    def __init__(self, list_paths, num_items, seq_len, batch_size, num_targets=-1, alpha=1., mask_prob=0.4, 
                 reverse_prob=0.2, get_session=False, get_only_first_on_val=False, seq_len_target=None,
                 min_size_seq_to_mask=2, is_val=False, is_test=False, avoid_repeats=False, shuffle=False, drop_remainder=False):
        self.list_paths = list_paths
        self.num_items = num_items
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_targets = num_targets
        self.alpha = alpha
        self.mask_prob = mask_prob
        self.reverse_prob = tf.constant(reverse_prob)
        self.shuffle = shuffle
        self.min_size_seq_to_mask = min_size_seq_to_mask
        self.avoid_repeats = avoid_repeats
        self.get_session = get_session
        self.seq_len_target = seq_len if not seq_len_target else seq_len_target
        self.get_only_first_on_val = get_only_first_on_val
        self.is_val = is_val
        self.is_test = is_test
        self.drop_remainder = drop_remainder

    def get_generator(self):
        dataset = tf.data.TFRecordDataset(self.list_paths, num_parallel_reads=AUTO, compression_type='GZIP')
        dataset = dataset.map(self.parse_tf_record, num_parallel_calls=AUTO)
        if self.is_val:
            dataset = dataset.map(self.make_transforms_val, num_parallel_calls=AUTO)
        elif self.is_test:
            dataset = dataset.map(self.make_transforms_test, num_parallel_calls=AUTO)
        else:
            dataset = dataset.map(self.make_transforms_train, num_parallel_calls=AUTO)
        
        dataset = dataset.map(self.set_shapes, num_parallel_calls=AUTO)
        # dataset = dataset.map(self.normalize_features, num_parallel_calls=AUTO)
        if self.shuffle:
            dataset = dataset.shuffle(self.batch_size*50, reshuffle_each_iteration=True)

        dataset = dataset.batch(self.batch_size, num_parallel_calls=AUTO, drop_remainder=self.drop_remainder).prefetch(AUTO)
        return dataset

    def parse_tf_record(self, data):
        features_context = {
             "session": tf.io.FixedLenFeature([], tf.int64),
             "size_session": tf.io.FixedLenFeature([], tf.int64),
        }
        if not self.is_val:
            features_seq = {
                "seq_aid" : tf.io.FixedLenSequenceFeature(shape=[1], dtype=tf.int64, allow_missing=False),
                "seq_type": tf.io.FixedLenSequenceFeature(shape=[1], dtype=tf.int64, allow_missing=False),
                "seq_qt_events": tf.io.FixedLenSequenceFeature(shape=[1], dtype=tf.int64, allow_missing=False),
                "seq_time_encoding": tf.io.FixedLenSequenceFeature(shape=[8], dtype=tf.float32, allow_missing=False),
                "seq_recency_aid": tf.io.FixedLenSequenceFeature(shape=[1], dtype=tf.float32, allow_missing=False)
            }
        else:
            features_seq = {
                "seq_aid" : tf.io.FixedLenSequenceFeature(shape=[1], dtype=tf.int64, allow_missing=False),
                "seq_type": tf.io.FixedLenSequenceFeature(shape=[1], dtype=tf.int64, allow_missing=False),
                "seq_qt_events": tf.io.FixedLenSequenceFeature(shape=[1], dtype=tf.int64, allow_missing=False),
                "seq_aid_target" : tf.io.FixedLenSequenceFeature(shape=[1], dtype=tf.int64, allow_missing=False),
                "seq_type_target": tf.io.FixedLenSequenceFeature(shape=[1], dtype=tf.int64, allow_missing=False),
                "seq_time_encoding": tf.io.FixedLenSequenceFeature(shape=[8], dtype=tf.float32, allow_missing=False),
                "seq_recency_aid": tf.io.FixedLenSequenceFeature(shape=[1], dtype=tf.float32, allow_missing=False)
            }
        data_context, data_sequence = tf.io.parse_single_sequence_example(data, context_features=features_context, sequence_features=features_seq)
        return data_context, data_sequence

    def pad_sequence(self, seq_to_pad, maxlen, return_pad_mask=False, dtype=tf.float32):
        length, num_feats = tf.shape(seq_to_pad)[0], tf.shape(seq_to_pad)[-1]
        ###
        if length < maxlen:
            pad = tf.zeros((maxlen - length, num_feats), dtype)
            seq = tf.concat([seq_to_pad, pad], axis=0)
            pad_mask = tf.concat([tf.ones(tf.shape(seq_to_pad), dtype=seq_to_pad.dtype), 
                                 pad], axis=0)
        else:
            seq = seq_to_pad[-maxlen:, :]
            pad_mask = tf.ones((maxlen, tf.shape(seq_to_pad)[-1]), dtype=seq_to_pad.dtype)
        if return_pad_mask:
            return seq, pad_mask
        return seq 

    def make_transforms_val(self, dict_context, dict_sequences):
        seq_items, seq_type, seq_time_encoding, seq_recency =  dict_sequences['seq_aid'], dict_sequences['seq_type'], dict_sequences['seq_time_encoding'], dict_sequences['seq_recency_aid']
        seq_qt_events = dict_sequences['seq_qt_events']
        seq_items_target_raw, seq_type_target_raw =  dict_sequences['seq_aid_target'], dict_sequences['seq_type_target']
        session, qt_size_seq = dict_context['session'], dict_context['size_session']
        seq_recency = self.normalize_features(seq_recency, stats=(3.904908, 1.763647))
        seq_qt_events = self.normalize_features(seq_qt_events, stats=(1.132127, 0.529691))
        ###
        # Build target
        seq_items, seq_target = seq_items, seq_items_target_raw[:1] if not self.get_session else seq_items_target_raw[:self.seq_len_target]
        seq_type, seq_type_target = seq_type, seq_type_target_raw[:1] if not self.get_session else seq_type_target_raw[:self.seq_len_target]
        seq_items_target = tf.concat([seq_items, seq_target], axis=0)
        seq_type_target = tf.concat([seq_type, seq_type_target], axis=0)
        ###
        #Mask last position
        seq_items = tf.concat([seq_items, tf.zeros((1, tf.shape(seq_items)[1]), tf.int64)], axis=0)
        seq_type = tf.concat([seq_type, seq_type_target[:1]], axis=0)
        seq_qt_events = tf.concat([seq_qt_events, tf.zeros((1, tf.shape(seq_qt_events)[1]), tf.float32)], axis=0)
        seq_time_encoding = tf.concat([seq_time_encoding, tf.zeros((1, tf.shape(seq_time_encoding)[1]), tf.float32)], axis=0)
        seq_recency = tf.concat([seq_recency, tf.zeros((1, tf.shape(seq_recency)[1]), tf.float32)], axis=0)
        ###
        idx_masked = tf.clip_by_value(tf.shape(seq_items)[0]-1, 0, self.seq_len-1)
        seq_items, _ = self.pad_sequence(seq_items, maxlen=self.seq_len, return_pad_mask=True, dtype=tf.int64)
        seq_type = self.pad_sequence(seq_type, maxlen=self.seq_len, return_pad_mask=False, dtype=tf.int64)
        seq_qt_events = self.pad_sequence(seq_qt_events, maxlen=self.seq_len, return_pad_mask=False, dtype=tf.float32)
        seq_time_encoding = self.pad_sequence(seq_time_encoding, maxlen=self.seq_len, return_pad_mask=False, dtype=tf.float32)  
        seq_recency = self.pad_sequence(seq_recency, maxlen=self.seq_len, return_pad_mask=False, dtype=tf.float32)  
        seq_items_target = self.pad_sequence(seq_items_target, maxlen=self.seq_len_target, return_pad_mask=False, dtype=tf.int64)  
        seq_type_target = self.pad_sequence(seq_type_target, maxlen=self.seq_len_target, return_pad_mask=False, dtype=tf.int64)
        
        if self.get_session:
            seq_items_target_all = self.pad_sequence(seq_items_target_raw[:self.seq_len_target], maxlen=self.seq_len_target, return_pad_mask=False, dtype=tf.int64)  
            seq_type_target_all = self.pad_sequence(seq_type_target_raw[:self.seq_len_target], maxlen=self.seq_len_target, return_pad_mask=False, dtype=tf.int64) 
            return (seq_items, seq_type, seq_time_encoding, seq_qt_events, seq_recency), (seq_items_target_all[:, 0], seq_type_target_all[:, 0], idx_masked), session

        return (seq_items, seq_type, seq_time_encoding, seq_qt_events, seq_recency), seq_items_target[:, 0]

    def make_transforms_test(self, dict_context, dict_sequences):
        seq_items, seq_type, seq_time_encoding, seq_recency = dict_sequences['seq_aid'], dict_sequences['seq_type'], dict_sequences['seq_time_encoding'], dict_sequences['seq_recency_aid']
        seq_qt_events = dict_sequences['seq_qt_events']
        session, qt_size_seq = dict_context['session'], dict_context['size_session']
        seq_recency = self.normalize_features(seq_recency, stats=(3.904908, 1.763647))
        seq_qt_events = self.normalize_features(seq_qt_events, stats=(1.132127, 0.529691))
        ###
        seq_items = seq_items[-self.seq_len:, :]
        seq_type = seq_type[-self.seq_len:, :]
        seq_qt_events = seq_qt_events[-self.seq_len:, :]
        seq_time_encoding = seq_time_encoding[-self.seq_len:, :]
        seq_recency = seq_recency[-self.seq_len:, :]
        idx_masked = tf.clip_by_value(tf.shape(seq_items)[0]-1, 0, self.seq_len-1)
        # Mask last position
        seq_items = tf.concat([seq_items, tf.zeros((1, tf.shape(seq_items)[1]), tf.int64)], axis=0)
        seq_type = tf.concat([seq_type, tf.zeros((1, tf.shape(seq_type)[1]), tf.int64)], axis=0)
        seq_qt_events = tf.concat([seq_qt_events, tf.zeros((1, tf.shape(seq_qt_events)[1]), tf.float32)], axis=0)
        seq_time_encoding = tf.concat([seq_time_encoding, tf.zeros((1, tf.shape(seq_time_encoding)[1]), tf.float32)], axis=0)
        seq_recency = tf.concat([seq_recency, tf.zeros((1, tf.shape(seq_recency)[1]), tf.float32)], axis=0)
        ###
        seq_items, _ = self.pad_sequence(seq_items, maxlen=self.seq_len, return_pad_mask=True, dtype=tf.int64)
        seq_type = self.pad_sequence(seq_type, maxlen=self.seq_len, return_pad_mask=False, dtype=tf.int64)
        seq_qt_events = self.pad_sequence(seq_qt_events, maxlen=self.seq_len, return_pad_mask=False, dtype=tf.float32)
        seq_time_encoding = self.pad_sequence(seq_time_encoding, maxlen=self.seq_len, return_pad_mask=False, dtype=tf.float32)   
        seq_recency = self.pad_sequence(seq_recency, maxlen=self.seq_len, return_pad_mask=False, dtype=tf.float32)   
        if self.get_session:
            return (seq_items, seq_type, seq_time_encoding, seq_qt_events, seq_recency), idx_masked, session

        return (seq_items, seq_type, seq_time_encoding, seq_qt_events, seq_recency), idx_masked

  
    def make_transforms_train(self, dict_context, dict_sequences):
        seq_items, seq_type, seq_time_encoding, seq_recency =  dict_sequences['seq_aid'], dict_sequences['seq_type'], dict_sequences['seq_time_encoding'], dict_sequences['seq_recency_aid']
        seq_qt_events = dict_sequences['seq_qt_events']
        qt_size_seq = dict_context['size_session']
        seq_recency = self.normalize_features(seq_recency, stats=(3.904908, 1.763647))
        seq_qt_events = self.normalize_features(seq_qt_events, stats=(1.132127, 0.529691))
        ### 
        # With prob reverse
        if tf.random.uniform(shape=(1,1)) <= self.reverse_prob:
            seq_items = tf.reverse(seq_items, axis=[0])
            seq_type = tf.reverse(seq_type, axis=[0])
            seq_qt_events = tf.reverse(seq_qt_events, axis=[0])
            seq_time_encoding = tf.reverse(seq_time_encoding, axis=[0])
            seq_recency = tf.reverse(seq_recency, axis=[0])
            
        # If our seq is longer than seq_len we can use it for data augmentation purpose 
        # and select a random idx to begin with.
        if tf.shape(seq_items)[0] > self.seq_len:
            idx_list = tf.range(tf.shape(seq_items)[0]-self.seq_len) 
            rand_idx = tf.random.shuffle(idx_list)[0]
            seq_items = seq_items[rand_idx:(rand_idx+self.seq_len), :]
            seq_type = seq_type[rand_idx:(rand_idx+self.seq_len), :]
            seq_qt_events = seq_qt_events[rand_idx:(rand_idx+self.seq_len), :]
            seq_time_encoding = seq_time_encoding[rand_idx:(rand_idx+self.seq_len), :]
            seq_recency = seq_recency[rand_idx:(rand_idx+self.seq_len), :]

        # Check if all items are the same
        uniques, idxs = tf.unique(seq_items[:, 0])
        if tf.shape(uniques)[0]==1 and tf.shape(seq_items)[0] >= 4:
            seq_items = tf.zeros(tf.shape(seq_items), seq_items.dtype)
        
        qt_size_seq = tf.shape(seq_items)[0]

        ## Get idxs to mask for inputs and targets
        idxs_inputs, idxs_target = self.mask_indexes(qt_size_seq)

        # Mask inputs and targets
        seq_items_raw = seq_items
        updates_items = tf.zeros((len(idxs_inputs), seq_items.shape[-1]), tf.int64)
        updates_qt_events = tf.zeros((len(idxs_inputs), seq_qt_events.shape[-1]), tf.float32) 
        updates_time_encoding = tf.zeros((len(idxs_inputs), seq_time_encoding.shape[-1]), tf.float32)
        updates_recency = tf.zeros((len(idxs_inputs), seq_recency.shape[-1]), tf.float32)
        updates_target = tf.zeros((len(idxs_target), seq_items_raw.shape[-1]), tf.int64)
        
        seq_items = tf.tensor_scatter_nd_update(seq_items, idxs_inputs, updates_items)
        seq_qt_events = tf.tensor_scatter_nd_update(seq_qt_events, idxs_inputs, updates_qt_events)
        seq_time_encoding = tf.tensor_scatter_nd_update(seq_time_encoding, idxs_inputs, updates_time_encoding)
        seq_recency = tf.tensor_scatter_nd_update(seq_recency, idxs_inputs, updates_recency)
        seq_target = tf.tensor_scatter_nd_update(seq_items_raw, idxs_target, updates_target)
        
        # Padding
        seq_items, pad_mask = self.pad_sequence(seq_items, maxlen=self.seq_len, return_pad_mask=True, dtype=tf.int64)
        seq_type = self.pad_sequence(seq_type, maxlen=self.seq_len, return_pad_mask=False, dtype=tf.int64)
        seq_qt_events = self.pad_sequence(seq_qt_events, maxlen=self.seq_len, return_pad_mask=False, dtype=tf.float32)
        seq_time_encoding = self.pad_sequence(seq_time_encoding, maxlen=self.seq_len, return_pad_mask=False, dtype=tf.float32) 
        seq_recency = self.pad_sequence(seq_recency, maxlen=self.seq_len, return_pad_mask=False, dtype=tf.float32)  
        seq_target = self.pad_sequence(seq_target, maxlen=self.seq_len, return_pad_mask=False, dtype=tf.int64)  

        return (seq_items, seq_type, seq_time_encoding, seq_qt_events, seq_recency), seq_target[:, 0]
  
    def normalize_features(self, features, stats):
        mean, std = stats
        features = tf.cast(features, tf.float32)
        return (features - tf.constant(mean)/tf.constant(std))

    def mask_indexes(self, qt_size_seq):
        num_to_mask = tf.clip_by_value(tf.cast(tf.round(tf.cast(qt_size_seq, tf.float32) * self.mask_prob), tf.int32), 1, self.seq_len-1)
        probs_recency = self.alpha ** (tf.cast(qt_size_seq, tf.float32)-tf.range(qt_size_seq, dtype=tf.float32))
        probs_rndm = tf.random.uniform(shape=(qt_size_seq,), minval=0, maxval=1)
        if self.alpha==1:
            idxs_inputs = tf.cast(tf.where(probs_rndm >= (1-self.mask_prob)), tf.int64) # -> we mask to zero the inputs as we dont want to leak 
            idxs_target = tf.cast(tf.where(probs_rndm < (1-self.mask_prob)), tf.int64) # -> we mask to zero the targets as the loss will only be applied on non zero

            # If all items are masked we leave an item unmasked
            if tf.cast(tf.shape(idxs_inputs)[0], tf.int64) == tf.cast(qt_size_seq, tf.int64):
                idxs_target = idxs_inputs[-1:]
                idxs_inputs = idxs_inputs[:-1]

            # If no item has been masked we leave at least one item masked(be careful of size=1 seqs)
            if tf.cast(tf.shape(idxs_inputs)[0], tf.int64) == tf.constant(0, dtype=tf.int64):
                all_idxs = tf.cast(tf.random.shuffle(tf.range(0, qt_size_seq)), dtype=tf.int64)
                idxs_inputs = all_idxs[:1][:, tf.newaxis]
                idxs_target = all_idxs[1:][:, tf.newaxis]
        else:
            probs = probs_recency * probs_rndm
            idxs = tf.argsort(probs, direction='DESCENDING')
            idxs_inputs = tf.cast(idxs[:num_to_mask][:, tf.newaxis], tf.int64)
            idxs_target = tf.cast(idxs[num_to_mask:][:, tf.newaxis], tf.int64)
            
        return idxs_inputs, idxs_target

    def set_shapes(self, features, targets=None, session=None):
        features[0].set_shape((self.seq_len, 1))
        features[1].set_shape((self.seq_len, 1))
        features[2].set_shape((self.seq_len, 8))
        features[3].set_shape((self.seq_len, 1))
        if self.get_session:
            return features, targets, session
        return features, targets


#################################################
#                   SASREC
################################################

class SASRecDataLoader:
    """
    Class that iterates over tfrecords in order to get the sequences.
    """
    def __init__(self, list_paths, num_items, seq_len, batch_size, num_targets=-1, mask_prob=0.4, 
                 reverse_prob=0.2, get_session=False, get_only_first_on_val=False, seq_len_target=None,
                 min_size_seq_to_mask=2, is_val=False, is_test=False, avoid_repeats=False, shuffle=False, drop_remainder=False):
        self.list_paths = list_paths
        self.num_items = num_items
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_targets = num_targets
        self.mask_prob = mask_prob
        self.reverse_prob = tf.constant(reverse_prob)
        self.shuffle = shuffle
        self.min_size_seq_to_mask = min_size_seq_to_mask
        self.avoid_repeats = avoid_repeats
        self.get_session = get_session
        self.seq_len_target = seq_len if not seq_len_target else seq_len_target
        self.get_only_first_on_val = get_only_first_on_val
        self.is_val = is_val
        self.is_test = is_test
        self.drop_remainder = drop_remainder

    def get_generator(self):
        dataset = tf.data.TFRecordDataset(self.list_paths, num_parallel_reads=AUTO, compression_type='GZIP')
        dataset = dataset.map(self.parse_tf_record, num_parallel_calls=AUTO)
        if self.is_val:
            dataset = dataset.map(self.make_transforms_val, num_parallel_calls=AUTO)
        elif self.is_test:
            dataset = dataset.map(self.make_transforms_test, num_parallel_calls=AUTO)
        else:
            dataset = dataset.map(self.make_transforms_train, num_parallel_calls=AUTO)
        
        dataset = dataset.map(self.set_shapes, num_parallel_calls=AUTO)
        # dataset = dataset.map(self.normalize_features, num_parallel_calls=AUTO)
        if self.shuffle:
            dataset = dataset.shuffle(self.batch_size*50, reshuffle_each_iteration=True)

        dataset = dataset.batch(self.batch_size, num_parallel_calls=AUTO, drop_remainder=self.drop_remainder).prefetch(AUTO)
        return dataset

    def parse_tf_record(self, data):
        features_context = {
             "session": tf.io.FixedLenFeature([], tf.int64),
             "size_session": tf.io.FixedLenFeature([], tf.int64),
        }
        if not self.is_val:
            features_seq = {
                "seq_aid" : tf.io.FixedLenSequenceFeature(shape=[1], dtype=tf.int64, allow_missing=False),
                "seq_type": tf.io.FixedLenSequenceFeature(shape=[1], dtype=tf.int64, allow_missing=False),
                "seq_time_encoding": tf.io.FixedLenSequenceFeature(shape=[8], dtype=tf.float32, allow_missing=False),
                "seq_recency_aid": tf.io.FixedLenSequenceFeature(shape=[1], dtype=tf.float32, allow_missing=False)
            }
        else:
            features_seq = {
                "seq_aid" : tf.io.FixedLenSequenceFeature(shape=[1], dtype=tf.int64, allow_missing=False),
                "seq_type": tf.io.FixedLenSequenceFeature(shape=[1], dtype=tf.int64, allow_missing=False),
                "seq_aid_target" : tf.io.FixedLenSequenceFeature(shape=[1], dtype=tf.int64, allow_missing=False),
                "seq_type_target": tf.io.FixedLenSequenceFeature(shape=[1], dtype=tf.int64, allow_missing=False),
                "seq_time_encoding": tf.io.FixedLenSequenceFeature(shape=[8], dtype=tf.float32, allow_missing=False),
                "seq_recency_aid": tf.io.FixedLenSequenceFeature(shape=[1], dtype=tf.float32, allow_missing=False)
            }
        data_context, data_sequence = tf.io.parse_single_sequence_example(data, context_features=features_context, sequence_features=features_seq)
        return data_context, data_sequence

    def pad_sequence(self, seq_to_pad, maxlen, return_pad_mask=False, dtype=tf.float32):
        length, num_feats = tf.shape(seq_to_pad)[0], tf.shape(seq_to_pad)[-1]
        ###
        if length < maxlen:
            pad = tf.zeros((maxlen - length, num_feats), dtype)
            seq = tf.concat([seq_to_pad, pad], axis=0)
            pad_mask = tf.concat([tf.ones(tf.shape(seq_to_pad), dtype=seq_to_pad.dtype), 
                                 pad], axis=0)
        else:
            seq = seq_to_pad[-maxlen:, :]
            pad_mask = tf.ones((maxlen, tf.shape(seq_to_pad)[-1]), dtype=seq_to_pad.dtype)
        if return_pad_mask:
            return seq, pad_mask
        return seq 

    def make_transforms_val(self, dict_context, dict_sequences):
        seq_items, seq_type, seq_time_encoding, seq_recency =  dict_sequences['seq_aid'], dict_sequences['seq_type'], dict_sequences['seq_time_encoding'], dict_sequences['seq_recency_aid']
        seq_items_target_raw, seq_type_target_raw =  dict_sequences['seq_aid_target'], dict_sequences['seq_type_target']
        session, qt_size_seq = dict_context['session'], dict_context['size_session']
        seq_recency = self.normalize_features(seq_recency)
        ###
        # Build target
        seq_items, seq_target = seq_items, seq_items_target_raw[:1] if not self.get_session else seq_items_target_raw[:self.seq_len_target]
        seq_type, seq_type_target = seq_type, seq_type_target_raw[:1] if not self.get_session else seq_type_target_raw[:self.seq_len_target]
        seq_items_target = tf.concat([seq_items, seq_target], axis=0)
        seq_type_target = tf.concat([seq_type, seq_type_target], axis=0)
        ###
        ###
        idx_masked = tf.clip_by_value(tf.shape(seq_items)[0]-1, 0, self.seq_len-1)
        seq_items, pad_mask = self.pad_sequence(seq_items, maxlen=self.seq_len, return_pad_mask=True, dtype=tf.int64)
        seq_type = self.pad_sequence(seq_type, maxlen=self.seq_len, return_pad_mask=False, dtype=tf.int64)
        seq_time_encoding = self.pad_sequence(seq_time_encoding, maxlen=self.seq_len, return_pad_mask=False, dtype=tf.float32)  
        seq_recency = self.pad_sequence(seq_recency, maxlen=self.seq_len, return_pad_mask=False, dtype=tf.float32)  
        seq_items_target = self.pad_sequence(seq_items_target, maxlen=self.seq_len_target, return_pad_mask=False, dtype=tf.int64)  
        seq_type_target = self.pad_sequence(seq_type_target, maxlen=self.seq_len_target, return_pad_mask=False, dtype=tf.int64)
        
        if self.get_session:
            seq_items_target_all = self.pad_sequence(seq_items_target_raw[:self.seq_len_target], maxlen=self.seq_len_target, return_pad_mask=False, dtype=tf.int64)  
            seq_type_target_all = self.pad_sequence(seq_type_target_raw[:self.seq_len_target], maxlen=self.seq_len_target, return_pad_mask=False, dtype=tf.int64) 
            return (seq_items, seq_type, seq_time_encoding, seq_recency), (seq_items_target_all[:, 0], seq_type_target_all[:, 0], idx_masked), session

        return (seq_items, seq_type, seq_time_encoding, seq_recency), seq_items_target[:, 0]

    def make_transforms_test(self, dict_context, dict_sequences):
        seq_items, seq_type, seq_time_encoding, seq_recency =  dict_sequences['seq_aid'], dict_sequences['seq_type'], dict_sequences['seq_time_encoding'], dict_sequences['seq_recency_aid']
        session, qt_size_seq = dict_context['session'], dict_context['size_session']
        seq_recency = self.normalize_features(seq_recency)
        ###
        seq_items = seq_items[-self.seq_len:, :]
        seq_type = seq_type[-self.seq_len:, :]
        seq_time_encoding = seq_time_encoding[-self.seq_len:, :]
        seq_recency = seq_recency[-self.seq_len:, :]
        # Padding
        ###
        idx_masked = tf.clip_by_value(tf.shape(seq_items)[0]-1, 0, self.seq_len-1)
        seq_items, pad_mask = self.pad_sequence(seq_items, maxlen=self.seq_len, return_pad_mask=True, dtype=tf.int64)
        seq_type = self.pad_sequence(seq_type, maxlen=self.seq_len, return_pad_mask=False, dtype=tf.int64)
        seq_time_encoding = self.pad_sequence(seq_time_encoding, maxlen=self.seq_len, return_pad_mask=False, dtype=tf.float32)   
        seq_recency = self.pad_sequence(seq_recency, maxlen=self.seq_len, return_pad_mask=False, dtype=tf.float32)   
        if self.get_session:
            return (seq_items, seq_type, seq_time_encoding, seq_recency), idx_masked, session

        return (seq_items, seq_type, seq_time_encoding, seq_recency), idx_masked

  
    def make_transforms_train(self, dict_context, dict_sequences):
        seq_items, seq_type, seq_time_encoding, seq_recency =  dict_sequences['seq_aid'], dict_sequences['seq_type'], dict_sequences['seq_time_encoding'], dict_sequences['seq_recency_aid']
        qt_size_seq = dict_context['size_session']
        seq_recency = self.normalize_features(seq_recency)
        ### 
        # With prob reverse
        if tf.random.uniform(shape=(1,1)) <= self.reverse_prob:
            seq_items = tf.reverse(seq_items, axis=[0])
            seq_type = tf.reverse(seq_type, axis=[0])
            seq_time_encoding = tf.reverse(seq_time_encoding, axis=[0])
            seq_recency = tf.reverse(seq_recency, axis=[0])
            
        # If our seq is longer than seq_len we can use it for data augmentation purpose 
        # and select a random idx to begin with.
        if tf.shape(seq_items)[0] > self.seq_len:
            idx_list = tf.range(tf.shape(seq_items)[0]-self.seq_len) 
            rand_idx = tf.random.shuffle(idx_list)[0]
            seq_items = seq_items[rand_idx:(rand_idx+self.seq_len), :]
            seq_type = seq_type[rand_idx:(rand_idx+self.seq_len), :]
            seq_time_encoding = seq_time_encoding[rand_idx:(rand_idx+self.seq_len), :]
            seq_recency = seq_recency[rand_idx:(rand_idx+self.seq_len), :]
        
        qt_size_seq = tf.shape(seq_items)[0]
        
        seq_items_input = seq_items[:-1, :]
        seq_items_target = seq_items[1:, :]
        seq_type_input = seq_type[:-1, :]
        seq_time_encoding_input = seq_time_encoding[:-1, :]
        seq_recency_input = seq_recency[:-1, :]
        
        # Padding
        seq_items, pad_mask = self.pad_sequence(seq_items_input, maxlen=self.seq_len, return_pad_mask=True, dtype=tf.int64)
        seq_type = self.pad_sequence(seq_type_input, maxlen=self.seq_len, return_pad_mask=False, dtype=tf.int64)
        seq_time_encoding = self.pad_sequence(seq_time_encoding_input, maxlen=self.seq_len, return_pad_mask=False, dtype=tf.float32) 
        seq_recency = self.pad_sequence(seq_recency_input, maxlen=self.seq_len, return_pad_mask=False, dtype=tf.float32)  
        seq_target = self.pad_sequence(seq_items_target, maxlen=self.seq_len, return_pad_mask=False, dtype=tf.int64)  

        return (seq_items, seq_type, seq_time_encoding, seq_recency), seq_target[:, 0]
  
    def normalize_features(self, features):
        return (features - tf.constant(5.45)/tf.constant(1.09))

    def set_shapes(self, features, targets=None, session=None):
        features[0].set_shape((self.seq_len, 1))
        features[1].set_shape((self.seq_len, 1))
        features[2].set_shape((self.seq_len, 8))
        features[3].set_shape((self.seq_len, 1))
        if self.get_session:
            return features, targets, session
        return features, targets