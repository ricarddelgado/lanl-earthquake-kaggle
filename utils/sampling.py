import numpy as np

def generate_segment_start_ids(sampling_method, segment_size, train, quakes=None):
    if sampling_method == 'uniform':
        # With this approach we obtain 4194 segments
        num_segments_training = int(np.floor(train.shape[0] / segment_size))
        segment_start_ids = [i * segment_size for i in range(num_segments_training)]
    elif sampling_method == 'uniform_no_jump':
        # With this approach we obtain 4178 segments (99.5% of 'uniform')
        already_sampled = np.full(train.shape[0], False)
        num_segments_training = int(np.floor(train.shape[0] / segment_size))
        time_to_failure_jumps = np.diff(train['time_to_failure'].values)
        num_good_segments_found = 0
        segment_start_ids = []
        for i in range(num_segments_training):
            idx = i * segment_size
            # Detect if there is a discontinuity on the time_to_failure signal within the segment
            if quakes is not None:
                bad_quakes = quakes.loc[quakes['valid'] == False]
                end_cond = (idx + segment_size) <= bad_quakes['end_idx']
                start_cond = idx >= bad_quakes['start_idx']
                cond = np.sum(start_cond * end_cond)
                if cond==0:
                    max_jump = np.max(time_to_failure_jumps[idx:idx + segment_size])
                    if max_jump < 5:
                        segment_start_ids.append(idx)
                        num_good_segments_found += 1
            else:
                max_jump = np.max(time_to_failure_jumps[idx:idx + segment_size])
                if max_jump < 5:
                    segment_start_ids.append(idx)
                    num_good_segments_found += 1
                    
        segment_start_ids.sort()
    elif sampling_method == 'random_no_jump':
        # With this approach we obtain 4194 segments
        num_segments_training = int(np.floor(train.shape[0] / segment_size)) #arbitrary choice
        time_to_failure_jumps = np.diff(train['time_to_failure'].values)
        num_good_segments_found = 0
        segment_start_ids = []
        while num_segments_training != num_good_segments_found:
            # Generate a random sampling position
            idx = random.randint(0, train.shape[0] - segment_size - 1)
            # Detect if there is a discontinuity on the time_to_failure signal within the segment
            max_jump = np.max(time_to_failure_jumps[idx:idx + segment_size])
            if max_jump < 5:
                segment_start_ids.append(idx)
                num_good_segments_found += 1
        segment_start_ids.sort()
    else:
        raise NameError('Method does not exist')
    return segment_start_ids