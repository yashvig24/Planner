import numpy as np

class Sampler:
    def __init__(self, env):
        self.env = env
        self.xlimit = self.env.xlimit
        self.ylimit = self.env.ylimit

    def sample(self, num_samples):
        """
        Samples configurations.
        Each configuration is (x, y).

        @param num_samples: Number of sample configurations to return
        @return 2D numpy array of size [num_samples x 2]
        """

        # Implement here
        samples = np.empty((0,2), int)

        vertices_to_add = num_samples
        while vertices_to_add != 0:
            random_vertices_x = np.random.uniform(self.xlimit[0], self.xlimit[1], (vertices_to_add, 1))
            random_vertices_y = np.random.uniform(self.ylimit[0], self.ylimit[1], (vertices_to_add, 1))
            random_vertices = np.append(random_vertices_x, random_vertices_y, axis = 1)
            random_vertices = np.floor(random_vertices)
            samples = np.append(samples, random_vertices, axis = 0)
            samples = np.unique(random_vertices, axis = 0)
            vertices_to_add -= len(samples)
        
        return samples.astype(int)
