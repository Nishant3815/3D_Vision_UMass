import pickle as p

def save_data(meshes, min_y, max_y, N, shape_labels, filename):
    """
    Save data to avoid reloading meshes each time you run the code.
    """
    data = {}
    data['meshes'] = meshes
    data['min_y'] = min_y
    data['max_y'] = max_y
    data['N'] = N
    data['shape_labels'] = shape_labels
    p.dump(data, open( filename, "wb" ) )

def load_data(filename):
    """
    Load data from saved pickle data.
    """
    try:
        data = p.load( open( filename, "rb" ) )
        return data['meshes'], data['min_y'], data['max_y'], data['N'], data['shape_labels']
    except IOError as e:
        print("Couldn't open file (%s)." % e)
        return [], [], [], [], []