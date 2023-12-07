

def adjusted_r2(r2_score, sample_size, num_features):
    return 1 - (1 - r2_score) * (sample_size - 1) / (sample_size - num_features - 1)
