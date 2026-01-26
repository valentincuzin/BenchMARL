import torch
from collections.abc import Sequence
from torchrl.envs import Transform
from tensordict.utils import (
    _unravel_key_to_tuple,
    _zip_strict,
    expand_as_right,
    NestedKey,
)
from tensordict import TensorDictBase, unravel_key
from torchrl.data.tensor_specs import Composite, TensorSpec
from torchrl.envs.transforms.utils import _get_reset
from copy import copy


class ExtractFrom(Transform):
    """
    Extract slices value from a selected tensor with in_key, then copy it in the tensordict with out_key
    Only tested on [('agents', 'observation')] for now
    """
    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = [('agents', 'observation')],
        out_keys: Sequence[NestedKey] | None = None,
        slices: slice | None = None,
        ):
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.slices = slices
        self.out_size = self.slices.stop - self.slices.start
        self._keys_checked = False

    @property
    def in_keys(self) -> Sequence[NestedKey]:
        in_keys = self.__dict__.get("_in_keys", None)
        if in_keys in (None, []):
            in_keys = [('agents', 'observation')]
            self._in_keys = in_keys
        return in_keys

    @in_keys.setter
    def in_keys(self, value):
        if value is not None:
            if isinstance(value, (str, tuple)):
                value = [value]
            value = [unravel_key(val) for val in value]
        self._in_keys = value

    @property
    def out_keys(self) -> Sequence[NestedKey]:
        out_keys = self.__dict__.get("_out_keys", None)
        if out_keys in (None, []):
            out_keys = [('agents', 'extract')]
            self._out_keys = out_keys
        return out_keys

    @out_keys.setter
    def out_keys(self, value):
        # we must access the private attribute because this check occurs before
        # the parent env is defined
        print(self._in_keys, value)
        if value is not None and len(self._in_keys) != len(value):
            raise ValueError(
                f"ExtractFrom expects the same number of input {self._in_keys} and output keys {value}"
            )
        if value is not None:
            if isinstance(value, (str, tuple)):
                value = [value]
            value = [unravel_key(val) for val in value]
        self._out_keys = value

    @property
    def reset_keys(self) -> Sequence[NestedKey]:
        reset_keys = self.__dict__.get("_reset_keys", None)
        if reset_keys is None:
            parent = self.parent
            if parent is None:
                raise TypeError(
                    "reset_keys not provided but parent env not found. "
                    "Make sure that the reset_keys are provided during "
                    "construction if the transform does not have a container env."
                )
            # let's try to match the reset keys with the in_keys.
            # We take the filtered reset keys, which are the only keys that really
            # matter when calling reset, and check that they match the in_keys root.
            reset_keys = parent._filtered_reset_keys
            if len(reset_keys) == 1:
                reset_keys = list(reset_keys) * len(self.in_keys)

            def _check_match(reset_keys, in_keys):
                # if this is called, the length of reset_keys and in_keys must match
                for reset_key, in_key in _zip_strict(reset_keys, in_keys):
                    # having _reset at the root and the observation_key ("agent", "observation") is allowed
                    # but having ("agent", "_reset") and "observation" isn't
                    if isinstance(reset_key, tuple) and isinstance(in_key, str):
                        return False
                    if (
                        isinstance(reset_key, tuple)
                        and isinstance(in_key, tuple)
                        and in_key[: (len(reset_key) - 1)] != reset_key[:-1]
                    ):
                        return False
                return True

            if not _check_match(reset_keys, self.in_keys):
                raise ValueError(
                    f"Could not match the env reset_keys {reset_keys} with the {type(self)} in_keys {self.in_keys}. "
                    f"Please provide the reset_keys manually. Reset entries can be "
                    f"non-unique and must be right-expandable to the shape of "
                    f"the input entries."
                )
            reset_keys = copy(reset_keys)
            self._reset_keys = reset_keys

        if not self._keys_checked and len(reset_keys) != len(self.in_keys):
            raise ValueError(
                f"Could not match the env reset_keys {reset_keys} with the in_keys {self.in_keys}. "
                "Please make sure that these have the same length."
            )
        self._keys_checked = True

        return reset_keys

    @reset_keys.setter
    def reset_keys(self, value):
        if value is not None:
            if isinstance(value, (str, tuple)):
                value = [value]
            value = [unravel_key(val) for val in value]
        self._reset_keys = value

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Resets episode."""
        for in_key, reset_key, out_key in _zip_strict(
            self.in_keys, self.reset_keys, self.out_keys
        ):
            _reset = _get_reset(reset_key, tensordict)
            value = tensordict.get(out_key, default=None)
            if value is None:
                value = torch.zeros(self.parent.observation_spec[in_key].shape[:-1] + torch.Size([self.out_size]))
            else:
                value = torch.where(expand_as_right(~_reset, value), value, 0.0)
            tensordict_reset.set(out_key, value)
        return tensordict_reset

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        """Updates the episode out with the step observation."""
        # Update episode out
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            if in_key in next_tensordict.keys(include_nested=True):
                observation = next_tensordict.get(in_key)
                extracted = observation[..., self.slices]
                next_tensordict.set(out_key, extracted)
            elif not self.missing_tolerance:
                raise KeyError(f"'{in_key}' not found in tensordict {tensordict}")
        return next_tensordict

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        state_spec = input_spec["full_state_spec"]
        if state_spec is None:
            state_spec = Composite(shape=input_spec.shape, device=input_spec.device)
        state_spec.update(self._generate_out_spec())
        input_spec["full_state_spec"] = state_spec
        return input_spec

    def _generate_out_spec(self) -> Composite:
        out_spec = Composite()
        obs_spec = self.parent.full_observation_spec
        observation_spec_keys = self.parent.observation_keys
        # Define episode specs for all out_keys
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            if (
                in_key in observation_spec_keys
            ):  # if this out_key has a corresponding key in obs_spec
                out_key = _unravel_key_to_tuple(out_key)
                temp_out_spec = out_spec
                temp_rew_spec = obs_spec
                for sub_key in out_key[:-1]:
                    if (
                        not isinstance(temp_rew_spec, Composite)
                        or sub_key not in temp_rew_spec.keys()
                    ):
                        break
                    if sub_key not in temp_out_spec.keys():
                        temp_out_spec[sub_key] = temp_rew_spec[
                            sub_key
                        ].empty()
                    temp_rew_spec = temp_rew_spec[sub_key]
                    temp_out_spec = temp_out_spec[sub_key]

                out_spec[out_key] = obs_spec[in_key].clone()
                out_spec[out_key].shape = out_spec[out_key].shape[:-1] + torch.Size([self.out_size])
                out_spec[out_key].space.low = out_spec[out_key].space.low[...,:self.out_size]
                out_spec[out_key].space.high = out_spec[out_key].space.high[...,:self.out_size]
            else:
                raise ValueError(
                    f"The in_key: {in_key} is not present in the observation spec {obs_spec}."
                )
        return out_spec

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        """Transforms the observation spec, adding the new keys generated by ExtractFrom."""
        if not isinstance(observation_spec, Composite):
            observation_spec = Composite(
                observation=observation_spec, shape=self.parent.batch_size
            )
        observation_spec.update(self._generate_out_spec())
        return observation_spec

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            observation = tensordict[in_key]
            extracted = observation[..., self.slices]
            tensordict.set(out_key, extracted)
        return tensordict