import numpy as np 

from a_star_algorithm import a_star_search
from utils import load_yaml
from constants import OBSTACLE

def create_map(row_obstacle, col_obstacle, num_rows, num_columns):
    grid = np.zeros((num_rows, num_columns))
    grid[row_obstacle, col_obstacle] = OBSTACLE
    return grid
    

if __name__ == "__main__":

    params = load_yaml("initial_state.yaml")
    grid = create_map(**params)
    src = [4, 0]
    dest = [0, 5]
    # Run the A* search algorithm
    a_star_search(grid, src, dest)
