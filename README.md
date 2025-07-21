# A*-algorithm

Finds optimal path between source to destination without passing through obstacles.

We start from source and saves to open_Set the closest neighbors that improves the cost of themselves or we does not been there in the past, we use priority queue for selecting the best node for each step.

If the algorithm reaches to a wall, then it starts to take least preferred nodes, until it succeeds.

The benefit against DFS or BFS, is that you does not need to pass through places that reduces the heuristic.


# BatchQueue
Create batch queue implemented by torch tensor.
We create batch queue for LIFO and FIFO.
```
    def __init__(self, dims: torch.Tensor):
        """
        dims:
        [batch_size, queue_length, num_features]
        """

    def dequeue(self, batch_indices) -> torch.Tensor:
        """
        batch_indices: (N, )
        return (N, num_features) 
            where N <= batch_size
        """
        pass
   
    def enqueue(self, values, batch_indices) -> None:
        """
        values: (N, T, num_features)
        batch_indices: (N,)
        large index is newest, e.g. large t index is newest.
        """
        pass

    def peek(self, location, batch_indices):
        """
        location: (N,)
        batch_indices: (N,)
        return (N, num_features)
        # return Oldest for location == 0, 
            return newest for location == 1

        """
        pass
```