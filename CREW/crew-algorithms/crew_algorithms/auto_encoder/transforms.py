import os
import warnings
from time import time

import torch
import torch.nn as nn
import wandb
import wandb.errors
from torchrl.data.tensor_specs import (
    CompositeSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec,
    ContinuousBox
)
from tensordict import TensorDictBase
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs.transforms.transforms import Compose, Resize, Transform, CenterCrop, ObservationTransform, _apply_to_composite
from torchvision.utils import save_image
from torchrl.envs.transforms.utils import (
    _get_reset,
    _set_missing_tolerance,
    check_finite,
)

from crew_algorithms.auto_encoder import Encoder, Decoder, AutoEncoder, Encoder_Nature


class _EncoderNet(ObservationTransform):
    # invertible = False

    def __init__(
        self,
        in_keys,
        out_keys,
        num_channels,
        env_name: str = "bowling",
        del_keys: bool = True,
    ):
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.outdim = 64
        self.encoder = Encoder_Nature(num_channels, self.outdim)
        self.encoder.eval()
        self.del_keys = del_keys
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.env_name = env_name

        self.ae = AutoEncoder(num_channels, self.outdim)
        self.ae.eval()

    def load_weights(self, env_name, version):
        """Loads the weights for the encoder.

        Args:
            env_name: Name of the environment to load weights for.
            version: Version of the model to load.
        """

        # model_artifact = wandb.use_artifact(
        #     f"auto-encoder" f"/auto-encoder-{env_name}-model:{version}",
        #     type="model",
        # )
        # model_artifact_dir = model_artifact.download()
        # model_path = os.path.join(model_artifact_dir, "encoder.pt")
        
        if self.env_name == 'bowling':
            state_dict = torch.load('crew_algorithms/auto_encoder/weights/encoder_bowling.pth')
            print('loaded bowling encoder')
        elif self.env_name == 'tetris':
            state_dict = torch.load('crew_algorithms/auto_encoder/weights/encoder_tetris.pth')
            print('loaded tetris encoder')
        else:
            return

        self.encoder.load_state_dict(state_dict)
        # self.ae.load_state_dict(torch.load('crew_algorithms/auto_encoder/weights/ae_bowling.pth'))

    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(self.device)
        # with torch.inference_mode():
        out = self.encoder(obs).detach()
        out = nn.Flatten()(out)

        # from torchvision.utils import save_image
        # from datetime import datetime
        # name = datetime.now().strftime("%H%M%S%f")[:-4]

    
        # save_image(obs, 'visualize/%s_obs.png' % name)
        # save_image(self.ae(obs), 'visualize/%s_rec.png' % name)
        return out

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        space = observation_spec.space
        if isinstance(space, ContinuousBox):
            space.low = self._apply_transform(space.low)
            space.high = self._apply_transform(space.high)
            observation_spec.shape = space.low.shape
        else:
            observation_spec.shape = self._apply_transform(
                torch.zeros(observation_spec.shape)
            ).shape
        return observation_spec


class EncoderTransform(Compose):
    def __init__(
        self,
        env_name: str,
        num_channels: int,
        in_keys: list[str] | None = None,
        out_keys: list[str] | None = None,
        version: str = "latest",
    ):
        """Transforms raw observations into a smaller dimension by using a
        pretrained Encoder.

        This Encoder transform uses the pretrained Encoder part of the
        AutoEncoder outlined here:
        https://www.nature.com/articles/s41598-020-77918-x.

        Args:
            env_name: The name of the environment to use the encoder for.
            num_channels: The number of channels that will be fed to the
                encoder.
            in_keys: The input keys to transform.
            out_keys: The keys where the output will be stored.
            version: The version of the encoder to use.
        """
        self._device = None
        self._dtype = None
        self.env_name = env_name
        in_keys = in_keys if in_keys is not None else ["observation"]
        out_keys = out_keys if out_keys is not None else ["encoder_vec"]
        self.version = version

        transforms = []

        # Encoder
        # crop = CenterCrop(180, 180, in_keys=in_keys)
        # resize = Resize(256, 256, in_keys=in_keys)
        # resize = Resize(100, 100, in_keys=in_keys)

        # transforms.append(crop)
        # transforms.append(resize)
        network = _EncoderNet(
            in_keys=in_keys, out_keys=out_keys, num_channels=num_channels, del_keys=True, env_name=env_name
        )

        network.load_weights(env_name, version)
        # try:
        #     network.load_weights(env_name, version)
        # except (wandb.errors.CommError, FileNotFoundError):
        #     warnings.warn(
        #         f"Unable to find autoencoder for {env_name} with "
        #         f"version {version}. Will run with uninitialized "
        #         "weights for the autoencoder."
        #     )
        transforms.append(network)

        super().__init__(*transforms)

        if self._device is not None:
            self.to(self._device)
        if self._dtype is not None:
            self.to(self._dtype)

    def to(self, dest: DEVICE_TYPING | torch.dtype):
        if isinstance(dest, torch.dtype):
            self._dtype = dest
        else:
            self._device = dest
        return super().to(dest)

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype
