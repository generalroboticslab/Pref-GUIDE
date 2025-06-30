import functools
from typing import Optional, Sequence, Union

import torch
from tensordict import TensorDictBase
from tensordict.utils import NestedKey
from torchrl.data.tensor_specs import CompositeSpec
from torchrl.envs.transforms import Transform


class IndexSelectTransform(Transform):
    def __init__(
        self,
        indicies: Sequence[Sequence[Union[torch.IntTensor, torch.LongTensor]]],
        dims: Sequence[Sequence[int]] = 0,
        in_keys: Optional[Sequence[NestedKey]] = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
        create_copy=False,
    ):
        self.indicies = indicies
        self.dims = dims
        self.create_copy = create_copy
        super().__init__(in_keys, out_keys)

        if "done" in self.in_keys and not self.create_copy:
            raise ValueError(
                "Removing 'done' is not allowed. Set `create_copy` to `True` "
                "to create a copy of the done state."
            )
        if "reward" in self.in_keys and not self.create_copy:
            raise ValueError(
                "Removing 'reward' is not allowed. Set `create_copy` to `True` "
                "to create a copy of the reward entry."
            )

        if len(self.dims) != len(self.indicies):
            raise ValueError(
                f"The number of dims ({len(self.dims)}) \
                should match the number of indicies ({len(self.indicies)})."
            )
        for i in range(len(self.dims)):
            if len(self.dims[i]) != len(self.indicies[i]):
                raise ValueError(
                    f"The number of dims at {i} ({len(self.dims[i])}) should \
                    match the number of indicies at {i} ({len(self.indicies[i])})."
                )
        if len(self.in_keys) != len(indicies):
            raise ValueError(
                f"The number of in_keys ({len(self.in_keys)}) \
                should match the number of indicies ({len(self.indicies)})."
            )

        self.num_split_keys = functools.reduce(
            lambda acc_num_keys, indicies: acc_num_keys + len(indicies),
            self.indicies,
            0,
        )
        if self.num_split_keys != len(self.out_keys):
            raise ValueError(
                f"The number of out_keys ({len(self.out_keys)}) should match \
                the total number of indicies provided ({len(self.num_split_keys)})."
            )

    def _apply_transform(self, tensordict: TensorDictBase) -> TensorDictBase:
        split_tensors = []
        for indicies, dim, in_key in zip(self.indicies, self.dims, self.in_keys):
            for d, index in zip(dim, indicies):
                split_tensor = torch.index_select(
                    tensordict[in_key], d, torch.tensor(index)
                )
                split_tensors.append(split_tensor)
        if not self.create_copy:
            for in_key in self.in_keys:
                del tensordict[in_key]
        for i, out_key in enumerate(self.out_keys):
            tensordict.set(out_key, split_tensors[i])

    # def index_select(self, in_spec, dim, index):
    #     specs = []
    #     for

    def transform_output_spec(self, output_spec: CompositeSpec) -> CompositeSpec:
        output_spec = output_spec.clone()
        i = 0
        special_keys = ("reward", "done")
        for in_key, dim, indicies in zip(self.in_keys, self.dims, self.indicies):
            for d, index in zip(dim, indicies):
                out_key = self.out_keys[i]
                in_key_tensor_spec = (
                    output_spec[f"_{in_key}_spec"]
                    if in_key in special_keys
                    else output_spec["_observation_spec"][in_key]
                )
                if in_key in special_keys:
                    output_spec["_observation_spec"][out_key] = torch.index_select(
                        output_spec[f"_{in_key}_spec"].clone(), d, index
                    )
                if out_key in special_keys:
                    output_spec[out_key] = torch.index_select(
                        in_key_tensor_spec.clone(), d, index
                    )
                else:
                    output_spec["_observation_spec"][out_key] = torch.index_select(
                        in_key_tensor_spec.clone(), d, index
                    )
                i += 1
            if not self.create_copy and in_key not in special_keys:
                del output_spec["_observation_spec"][in_key]
        return output_spec
