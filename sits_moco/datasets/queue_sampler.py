"""
Custom sampler for queue-based sampling without replacement.
Ensures all municipalities are seen before repeats, with no boundary overlap.
"""

import random
from collections import deque
from typing import Iterator, Optional

import torch
from torch.utils.data import Sampler


class QueueSampler(Sampler):
    """
    Samples indices without replacement from a queue.
    When queue is empty, creates a new shuffled queue ensuring no boundary overlap.

    Args:
        data_source: Dataset to sample from
        sample_ratio: Fraction of total data to use per epoch (e.g., 0.2 = 20%)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        data_source,
        sample_ratio: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.sample_ratio = sample_ratio
        self.seed = seed

        # Calculate samples per epoch
        self.total_size = len(data_source)
        self.samples_per_epoch = max(1, int(self.total_size * sample_ratio))

        # Initialize queue and tracking
        self.queue = deque()
        self.last_items = []  # Track last items to avoid boundary overlap
        self.epoch = 0

        # Initialize random state
        if seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random()

        # Fill initial queue
        self._refill_queue()

    def _refill_queue(self):
        """Refill queue with shuffled indices, avoiding boundary overlap."""
        # Create list of all indices
        all_indices = list(range(self.total_size))

        # Remove last items from previous queue to avoid boundary overlap
        if len(self.last_items) > 0:
            remaining_indices = [i for i in all_indices if i not in self.last_items]
            # If we removed too many, just shuffle everything (edge case)
            if len(remaining_indices) < self.total_size // 2:
                remaining_indices = all_indices
        else:
            remaining_indices = all_indices

        # Shuffle remaining indices
        shuffled = remaining_indices.copy()
        self.rng.shuffle(shuffled)

        # Add to queue
        self.queue.extend(shuffled)

        # Update last_items for next refill (keep last few items)
        # Use min of samples_per_epoch or 10% of total, whichever is smaller
        num_to_track = min(self.samples_per_epoch, max(1, self.total_size // 10))
        self.last_items = (
            shuffled[-num_to_track:]
            if len(shuffled) >= num_to_track
            else shuffled.copy()
        )

    def __iter__(self) -> Iterator[int]:
        """Generate indices for one epoch."""
        sampled_indices = []
        samples_needed = self.samples_per_epoch

        while samples_needed > 0:
            # Check if queue has enough items
            if len(self.queue) >= samples_needed:
                # Take exactly what we need
                for _ in range(samples_needed):
                    sampled_indices.append(self.queue.popleft())
                samples_needed = 0
            else:
                # Take all remaining items from queue
                items_taken_this_iteration = len(self.queue)
                while len(self.queue) > 0:
                    sampled_indices.append(self.queue.popleft())
                samples_needed -= items_taken_this_iteration

                # Refill queue (ensures no boundary overlap)
                self._refill_queue()

                # Continue taking from new queue
                continue

        return iter(sampled_indices)

    def __len__(self) -> int:
        """Return number of samples per epoch."""
        return self.samples_per_epoch

    def set_epoch(self, epoch: int):
        """Set epoch for reproducibility."""
        self.epoch = epoch
        # Use epoch as additional seed component for different shuffles each epoch
        if self.seed is not None:
            self.rng = random.Random(self.seed + epoch)
        else:
            self.rng = random.Random()
