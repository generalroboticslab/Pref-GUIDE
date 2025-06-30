import torch
from tensordict.nn import TensorDictModule, TensorDictModuleWrapper
from tensordict.tensordict import TensorDictBase
from tensordict.utils import NestedKey


class ImitationLearningWrapper(TensorDictModuleWrapper):
    def __init__(
        self,
        policy: TensorDictModule,
        *,
        action_key: NestedKey | None = "action",
        il_enabled_key: NestedKey | None = "il_enabled",
        il_action_key: NestedKey | None = "il_action",
    ):
        super().__init__(policy)
        self.action_key = action_key
        self.il_enabled_key = il_enabled_key
        self.il_action_key = il_action_key

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = self.td_module.forward(tensordict)
        if isinstance(self.action_key, tuple) and len(self.action_key) > 1:
            action_tensordict = tensordict.get(self.action_key[:-1])
            action_key = self.action_key[-1]
        else:
            action_tensordict = tensordict
            action_key = self.action_key

        out = action_tensordict.get(action_key)

        # FIXME: In the future use il_enabled and il_action keys.
        #handle stack
        # print(tensordict[self.il_enabled_key].shape)
        cond = tensordict[self.il_enabled_key][0, -1, 1].item()
        il_action = tensordict[self.il_action_key][0, -1, 2].int().item()

        out = cond * il_action + (1 - cond) * out
        action_tensordict.set(action_key, out.to(torch.int64))
        return tensordict
