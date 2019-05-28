import numpy as np
import math
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
        @param end_configs: list of tuples of end configs
        @return numpy array of distances
        """
        # Implement here
        start_config[2] = math.radians(start_config[2])
        end_configs[:, 2] = math.radians(end_configs[:, 2])
        px, py, pyaw, distances = dubins_path_planning(np.tile(np.array(start_config), len(end_configs)), np.array(end_configs))
        return distances

    def compute_heuristic(self, config, goal):
        """
        Use the Dubins path length from config to goal as the heuristic distance.
        """
        # Implement here
        px, py, pyaw, heuristic = dubins_path_planning(config, goal, self.curvature)
        return heuristic

    def generate_path(self, config1, config2):
        """
        Generate a dubins path from config1 to config2
        The generated path is not guaranteed to be collision free
        Use dubins_path_planning to get a path
        return: (numpy array of [x, y, yaw], curve length)
        """
        # Implement here
        config1[2] = math.radians(config1[2])
        config2[2] = math.radians(config2[2])
        px, py, pyaw, clen = dubins_path_planning(config1, config2, self.curvature)
        path = np.array([px, py, pyaw]).transpose()
        #print(path, clen)
        return path, clen
