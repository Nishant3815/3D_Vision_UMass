import trimesh
import os


class Mesh():
    def __init__(self, train_dir, filename, number_of_bins):
        self.V = trimesh.load(os.path.join(train_dir, filename)).vertices
        self.name = filename