"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process, get_context
import torch.nn.functional as F
import time

from timeit import default_timer as timer


# def run(rank, size):
#     """ Distributed function to be implemented later. """
#     ### run each gym environment separately, computing get batch of size_T, experience
#     ### compute advantages and everything that is needed for the 

# get_context()
#     pass
# """Blocking point-to-point communication."""

# def run(rank, size):
#     tensor = torch.zeros(1).cuda()
#     if rank == 0:
#         tensor += 1
#         # Send the tensor to process 1
#         dist.send(tensor=tensor, dst=1)
#     else:
#         # Receive tensor from process 0
#         dist.recv(tensor=tensor, src=0)
#     print('Rank ', rank, ' has data ', tensor[0])

# Globals

# class Policy(nn.Module):
#     def __init__(self):
#         super(Policy, self).__init__()
#         self.affine1 = nn.Linear(4, 128)
#         self.action_head = nn.Linear(128, 2)
#         self.value_head = nn.Linear(128, 1)

#         self.saved_actions = []
#         self.rewards = []

#     def forward(self, x, only_value=False):
#         if only_value:
#             with torch.no_grad():
#                 x = F.relu(self.affine1(x))
#                 state_values = self.value_head(x)
#                 return state_values

#         x = F.relu(self.affine1(x))
#         action_scores = self.action_head(x)
#         state_values = self.value_head(x)
#         return F.softmax(action_scores, dim=-1), state_values

# """ All-Reduce example."""
def run(rank, size):
    # torch.cuda.set_device(0)
    """ Simple point-to-point communication. """
    # print('size', )
    if rank == 0:
        start = timer()

    group = dist.new_group([i for i in range(size)])
    tensor = torch.ones(1)
    if rank != 0:
        time.sleep(rank**2)
        tensor[0] = rank
    # gather_list = None
    # print(f"ranK: {rank}")
    # if rank == 0:
    gather_list = [torch.zeros_like(tensor) for i in range(size)]
    gather_list = [] if rank != 0 else gather_list

    dist.gather(tensor, gather_list, 0, group)
    # else:
        # dist.gather(tensor, dst=0)
    if rank == 0:
        # ...
        end = timer()
        print(end - start)
    print('Rank ', rank, ' has data ', gather_list)

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    size = 5
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()