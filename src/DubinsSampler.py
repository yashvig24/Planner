import numpy as np

class DubinsSampler:
    def __init__(self, env):
        self.env = env
        self.xlimit = self.env.xlimit
        self.ylimit = self.env.ylimit

    def sample(self, num_samples):
        """
        Samples configurations.
        Each configuration is (x, y, angle).

        @param num_samples: Number of sample configurations to return
        @return 2D numpy array of size [num_samples x 3]
        """
        samples = np.zeros((num_samples, 3))
        samples[:, 0] = np.random.uniform(self.xlimit[0], self.xlimit[1], num_samples)
        samples[:, 1] = np.random.uniform(self.ylimit[0], self.ylimit[1], num_samples)
        samples[:, 2] = np.random.uniform(-0.3, 0.3, num_samples) #correct max/min headings?

        # Implement here
        return samples
