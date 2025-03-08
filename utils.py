def background_class_for_granularity(granularity):
    return {"whole": 158, "part": 40, "subpart": 0}[granularity]

def num_labels_for_granularity(granularity):
    return {"whole": 159, "part": 41, "subpart": 204}[granularity]  # classes + 1 background
