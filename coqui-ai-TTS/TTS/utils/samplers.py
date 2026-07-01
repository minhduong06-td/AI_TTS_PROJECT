import math
import random
from collections.abc import Callable
from torch.utils.data.sampler import BatchSampler, Sampler, SubsetRandomSampler


class SubsetSampler(Sampler):
    def __init__(self, indices):
        super().__init__()
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class PerfectBatchSampler(Sampler):
    def __init__(
        self,
        dataset_items,
        classes,
        batch_size,
        num_classes_in_batch,
        num_gpus=1,
        shuffle=True,
        drop_last=False,
        label_key="class_name",
    ):
        super().__init__()
        assert batch_size % (num_classes_in_batch * num_gpus) == 0, (
            "Batch size must be divisible by number of classes times the number of data parallel devices (if enabled)."
        )

        label_indices = {}
        for idx, item in enumerate(dataset_items):
            label = item[label_key]
            if label not in label_indices.keys():
                label_indices[label] = [idx]
            else:
                label_indices[label].append(idx)

        if shuffle:
            self._samplers = [SubsetRandomSampler(label_indices[key]) for key in classes]
        else:
            self._samplers = [SubsetSampler(label_indices[key]) for key in classes]

        self._batch_size = batch_size
        self._drop_last = drop_last
        self._dp_devices = num_gpus
        self._num_classes_in_batch = num_classes_in_batch

    def __iter__(self):
        batch = []
        if self._num_classes_in_batch != len(self._samplers):
            valid_samplers_idx = random.sample(range(len(self._samplers)), self._num_classes_in_batch)
        else:
            valid_samplers_idx = None

        iters = [iter(s) for s in self._samplers]
        done = False

        while True:
            b = []
            for i, it in enumerate(iters):
                if valid_samplers_idx is not None and i not in valid_samplers_idx:
                    continue
                idx = next(it, None)
                if idx is None:
                    done = True
                    break
                b.append(idx)
            if done:
                break
            batch += b
            if len(batch) == self._batch_size:
                yield batch
                batch = []
                if valid_samplers_idx is not None:
                    valid_samplers_idx = random.sample(range(len(self._samplers)), self._num_classes_in_batch)

        if not self._drop_last:
            if len(batch) > 0:
                groups = len(batch) // self._num_classes_in_batch
                if groups % self._dp_devices == 0:
                    yield batch
                else:
                    batch = batch[: (groups // self._dp_devices) * self._dp_devices * self._num_classes_in_batch]
                    if len(batch) > 0:
                        yield batch

    def __len__(self):
        class_batch_size = self._batch_size // self._num_classes_in_batch
        return min(((len(s) + class_batch_size - 1) // class_batch_size) for s in self._samplers)


def identity(x):
    return x


class SortedSampler(Sampler):
    def __init__(self, data, sort_key: Callable = identity):
        super().__init__()
        self.data = data
        self.sort_key = sort_key
        zip_ = [(i, self.sort_key(row)) for i, row in enumerate(self.data)]
        zip_ = sorted(zip_, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip_]

    def __iter__(self):
        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.data)


class BucketBatchSampler(BatchSampler):
    def __init__(
        self,
        sampler,
        data,
        batch_size,
        drop_last,
        sort_key: Callable | list = identity,
        bucket_size_multiplier=100,
    ):
        super().__init__(sampler, batch_size, drop_last)
        self.data = data
        self.sort_key = sort_key
        _bucket_size = batch_size * bucket_size_multiplier
        if hasattr(sampler, "__len__"):
            _bucket_size = min(_bucket_size, len(sampler))
        self.bucket_sampler = BatchSampler(sampler, _bucket_size, False)

    def __iter__(self):
        for idxs in self.bucket_sampler:
            bucket_data = [self.data[idx] for idx in idxs]
            sorted_sampler = SortedSampler(bucket_data, self.sort_key)
            for batch_idx in SubsetRandomSampler(list(BatchSampler(sorted_sampler, self.batch_size, self.drop_last))):
                sorted_idxs = [idxs[i] for i in batch_idx]
                yield sorted_idxs

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return math.ceil(len(self.sampler) / self.batch_size)
