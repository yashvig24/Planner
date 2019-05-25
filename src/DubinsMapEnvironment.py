import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from Dubins import dubins_path_planning
from MapEnvironment import MapEnvironment

class DubinsMapEnvironment(MapEnvironment):

    def __init__(self, map_data, curvature=5):
        super(DubinsMapEnvironment, self).__init__(map_data)
        self.curvature = curvature

    def compute_distances(self, start_config, end_configs):
        """
        Compute distance from start_config and end_configs using Dubins path
        @param start_config: tuple of start config
        @param end_configs: list of tuples of end confings
        @return numpy array of distances
        """

        # Implement here

        return distances

    def compute_heuristic(self, config, goal):
        """
        Use the Dubins path length from config to goal as the heuristic distance.
        """

        # Implement here

        return heuristic

    def generate_path(self, config1, config2):
        """
        Generate a dubins path from config1 to config2
        The generated path is not guaranteed to be collision free
        Use dubins_path_planning to get a path
        return: (numpy array of [x, y, yaw], curve length)
        """

        # Implement here

        return path, clen