from tensordict.nn import TensorDictModule, TensorDictModuleWrapper
from tensordict.tensordict import TensorDictBase
from tensordict.utils import NestedKey


class ImitationLearningWrapper(TensorDictModuleWrapper):
    def __init__(
        self,
        policy: TensorDictModule,
        *,
        action_spec,
        action_key: NestedKey | None = "action",
        il_enabled_key: NestedKey | None = "il_enabled",
        il_action_key: NestedKey | None = "il_action",
    ):
        super().__init__(policy)
        self.action_spec = action_spec
        self.action_key = action_key
        self.il_enabled_key = il_enabled_key
        self.il_action_key = il_action_key

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict.set(self.action_key, self.action_spec.rand())
        if isinstance(self.action_key, tuple) and len(self.action_key) > 1:
            action_tensordict = tensordict.get(self.action_key[:-1])
            action_key = self.action_key[-1]
        else:
            action_tensordict = tensordict
            action_key = self.action_key

        out = action_tensordict.get(action_key)

        # FIXME: In the future use il_enabled and il_action keys.
        cond = tensordict[self.il_enabled_key][0][-2].item()
        il_action = tensordict[self.il_action_key][0][-1].item()

        out = cond * il_action + (1 - cond) * out
        action_tensordict.set(action_key, out)
        return tensordict
