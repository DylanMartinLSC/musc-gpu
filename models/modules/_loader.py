"""Bounded ordered loading without changing the dataset's item contract."""
from concurrent.futures import ThreadPoolExecutor

from torch.utils.data._utils.collate import default_collate
from torch.utils.data._utils.pin_memory import pin_memory as pin_batch


class ThreadedDatasetLoader:
    def __init__(self, dataset, batch_size=4, workers=4, prefetch_batches=2,
                 pin_memory=True):
        if batch_size < 1 or workers < 1 or prefetch_batches < 1:
            raise ValueError('Loader sizes must be positive')
        self.dataset = dataset
        self.batch_size = batch_size
        self.workers = workers
        self.prefetch_batches = prefetch_batches
        self.pin_memory = pin_memory

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # Futures remain in dataset order. Worker exceptions propagate to the
        # caller, and executor teardown also runs when iteration is interrupted.
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            indices = iter(range(len(self.dataset)))
            pending = []
            for _ in range(self.batch_size * self.prefetch_batches):
                index = next(indices, None)
                if index is None:
                    break
                pending.append(executor.submit(self.dataset.__getitem__, index))
            while pending:
                take = min(self.batch_size, len(pending))
                samples = [future.result() for future in pending[:take]]
                del pending[:take]
                for _ in range(take):
                    index = next(indices, None)
                    if index is None:
                        break
                    pending.append(executor.submit(self.dataset.__getitem__, index))
                batch = default_collate(samples)
                yield pin_batch(batch) if self.pin_memory else batch
