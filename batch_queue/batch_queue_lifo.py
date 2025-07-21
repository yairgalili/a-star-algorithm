import torch


class BatchQueue():
    """
    The oldest data: [:, 0, :]
    The newest data: [:, self.queue_lengths-1, :]
    """
    def __init__(self, dims: torch.Tensor):
        """
        dims:
        [batch_size, queue_length, num_features]
        """
        self.data = torch.zeros(dims.tolist())
        self.queue_lengths = torch.zeros(dims[0])
    
    def dequeue(self, batch_indices):
        self.queue_lengths[batch_indices] -= 1
        first_indices = self.queue_lengths[batch_indices]
        return self.data[batch_indices.to(int), first_indices.to(int), :]

    def enqueue(self, values, batch_indices):
        """
        values: (N, T, F)
        batch_indices: (N,)
        large index is newest, e.g. large t index is newest.
        """
        first_indices = torch.unsqueeze(self.queue_lengths[batch_indices], 1)
        indices_save = torch.unsqueeze(torch.arange(values.shape[1]), 0) + first_indices
        self.queue_lengths[batch_indices] += values.shape[1] 
        
        self.data[batch_indices.repeat(values.shape[1], 1).T.to(int), indices_save.to(int), :] = values 

    def peek(self, location, batch_indices):
        """
        location: (N,)
        batch_indices: (N,)
        """
        relevant_indices = self.queue_lengths[batch_indices] - 1
        # Oldest if location == 0
        relevant_indices[location == 0] = 0
        return self.data[batch_indices.to(int), relevant_indices.to(int), :]

if __name__ == "__main__":
    num_features = 3
    batch_indices = torch.tensor([1, 2])
    num_batches = batch_indices.shape[0]
    queue = BatchQueue(dims=torch.tensor([4, 100, num_features]))

    # values = (torch.arange(1, num_batches+1).unsqueeze(1) * torch.arange(1, 8).unsqueeze(0)).unsqueeze(2).repeat(1, 1, num_features)
    values = torch.rand(2, 7, 3)
    queue.enqueue(values=values, batch_indices=batch_indices)
    # queue.enqueue(values=values.to(torch.float32), batch_indices=batch_indices)
    newest = queue.dequeue(batch_indices=batch_indices)
    head_tail = queue.peek(location=torch.tensor([0, 1]), batch_indices=batch_indices)

