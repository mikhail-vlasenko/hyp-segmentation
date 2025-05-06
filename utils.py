class DoRemapObjects:
    value = False

def background_class_for_granularity(granularity):
    return {"whole": 11 if DoRemapObjects.value else 158, "part": 40, "subpart": 0}[granularity]

def num_labels_for_granularity(granularity):
    return {"whole": 12 if DoRemapObjects.value else 159, "part": 41, "subpart": 204}[granularity]  # classes + 1 background
