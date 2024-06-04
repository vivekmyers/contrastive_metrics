# coding=utf-8
# Copyright 2023 The Google Research Authors.
# Copyright Chongyi Zheng.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Metric Distillation networks definition."""

import dataclasses
from typing import Optional

from acme.jax import networks as networks_lib
from acme.jax import utils
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass
class MetricDistillationNetworks:
    """Network and pure functions for the Metric Distillation agent."""

    contrastive_network: networks_lib.FeedForwardNetwork
    quasimetric_network: networks_lib.FeedForwardNetwork
    critic_network: networks_lib.FeedForwardNetwork
    policy_network: networks_lib.FeedForwardNetwork
    log_prob: networks_lib.LogProbFn
    sample: networks_lib.SampleFn
    potential_network: Optional[networks_lib.FeedForwardNetwork] = None
    sample_eval: Optional[networks_lib.SampleFn] = None


def apply_policy_and_sample(networks, eval_mode=False):
    """Returns a function that computes actions."""
    sample_fn = networks.sample if not eval_mode else networks.sample_eval
    if not sample_fn:
        raise ValueError("sample function is not provided")

    def apply_and_sample(params, key, obs):
        return sample_fn(networks.policy_network.apply(params, obs), key)

    return apply_and_sample


def make_networks(
    spec,
    obs_dim,
    goal_to_obs,
    repr_dim=64,
    repr_norm=False,
    repr_norm_temp=1.0,
    adaptive_repr_norm_temp=True,
    hidden_layer_sizes=(256, 256),
    actor_min_std=1e-6,
    twin_q=False,
    use_image_obs=False,
    quasimetric="iqe",
    quasimetric_hidden_dim=256,
    quasimetric_num_groups=32,
):
    """Creates networks used by the agent."""

    num_dimensions = np.prod(spec.actions.shape, dtype=int)
    TORSO = networks_lib.AtariTorso  # pylint: disable=invalid-name

    assert quasimetric_hidden_dim % quasimetric_num_groups == 0

    def _unflatten_img(img):
        img = jnp.reshape(img, (-1, 64, 64, 3)) / 255.0
        return img

    def _repr_fn(obs, action, goal):
        # The optional input hidden is the image representations. We include this
        # as an input for the second Q value when twin_q = True, so that the two Q
        # values use the same underlying image representation.
        if use_image_obs:
            obs = _unflatten_img(obs)
            goal = _unflatten_img(goal)
            img_encoder = TORSO()
            state = img_encoder(obs)
            goal = img_encoder(goal)
        else:
            state = obs
            goal = goal

        sa_encoder = hk.nets.MLP(
            list(hidden_layer_sizes) + [repr_dim],
            w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
            activation=jax.nn.relu,
            name="sa_encoder",
        )
        sa_repr = sa_encoder(jnp.concatenate([state, action], axis=-1))

        if twin_q:
            sa_encoder2 = hk.nets.MLP(
                list(hidden_layer_sizes) + [repr_dim],
                w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                activation=jax.nn.relu,
                name="sa_encoder2",
            )
            sa_repr2 = sa_encoder2(state)
            sa_repr = jnp.stack([sa_repr, sa_repr2], axis=-1)
        else:
            sa_repr = sa_repr[..., None]

        g_encoder = hk.nets.MLP(
            list(hidden_layer_sizes) + [repr_dim],
            w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
            activation=jax.nn.relu,
            name="g_encoder",
        )
        g_repr = g_encoder(goal)

        if twin_q:
            g_encoder2 = hk.nets.MLP(
                list(hidden_layer_sizes) + [repr_dim],
                w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                activation=jax.nn.relu,
                name="g_encoder2",
            )
            g_repr2 = g_encoder2(goal)
            g_repr = jnp.stack([g_repr, g_repr2], axis=-1)
        else:
            g_repr = g_repr[..., None]

        if repr_norm:
            sa_repr = sa_repr / (jnp.linalg.norm(sa_repr, axis=1, keepdims=True) + 1e-8)
            g_repr = g_repr / (jnp.linalg.norm(g_repr, axis=1, keepdims=True) + 1e-8)

            if adaptive_repr_norm_temp:
                log_repr_norm_temp = hk.get_parameter(
                    "log_repr_norm_temp", [], dtype=sa_repr.dtype, init=jnp.zeros
                )
                sa_repr = sa_repr / jnp.exp(log_repr_norm_temp)
            else:
                sa_repr = sa_repr / repr_norm_temp

        return sa_repr, g_repr

    def _combine_repr(sa_repr, g_repr):
        return jnp.einsum("ikl,jkl->ijl", sa_repr, g_repr)

    def _contrastive_fn(obs, action, goal):
        sa_repr, g_repr = _repr_fn(obs, action, goal)
        outer = _combine_repr(sa_repr, g_repr)

        return outer, sa_repr, g_repr

    def _quasimetric_fn(obs, action, goal, dummy_action=None, reentrant=True):
        obs_, action_, goal_ = obs, action, goal
        if use_image_obs:
            obs = _unflatten_img(obs)
            goal = _unflatten_img(goal)
            img_encoder = TORSO()
            state = img_encoder(obs)
            goal = img_encoder(goal)
        else:
            state = jnp.concatenate([obs, action], axis=-1)
            if dummy_action is None:
                dummy_action = jnp.zeros((*goal.shape[:-1], action.shape[-1]))
            goal = jnp.concatenate([goal, dummy_action], axis=-1)

        if quasimetric == "iqe":
            alpha = hk.get_parameter("alpha", shape=(1,), init=jnp.zeros)
            alpha = jax.nn.sigmoid(alpha)

            encoder = hk.nets.MLP(
                list(hidden_layer_sizes) + [quasimetric_hidden_dim],
                w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                activation=jax.nn.relu,
                name="encoder",
            )
            x = encoder(state)
            y = encoder(goal)

            reshape = (
                quasimetric_num_groups,
                quasimetric_hidden_dim // quasimetric_num_groups,
            )
            x = jnp.reshape(x, (*x.shape[:-1], *reshape))
            y = jnp.reshape(y, (*y.shape[:-1], *reshape))
            valid = x < y
            D = x.shape[-1]
            xy = jnp.concatenate(jnp.broadcast_arrays(x, y), axis=-1)
            ixy = xy.argsort(axis=-1)
            # sxy = jnp.take_along_axis(xy, ixy, axis=-1)
            sxy = xy.sort(axis=-1)
            neg_inc_copies = jnp.take_along_axis(valid, ixy % D, axis=-1) * jnp.where(
                ixy < D, -1, 1
            )
            neg_inp_copies = jnp.cumsum(neg_inc_copies, axis=-1)
            neg_f = (neg_inp_copies < 0) * (-1.0)
            neg_incf = jnp.concatenate(
                [neg_f[..., :1], neg_f[..., 1:] - neg_f[..., :-1]], axis=-1
            )
            components = (sxy * neg_incf).sum(-1)
            dist = alpha * components.mean(axis=-1) + (1 - alpha) * components.max(
                axis=-1
            )

        elif quasimetric == "mrn":
            encoder = hk.nets.MLP(
                list(hidden_layer_sizes) + [2 * quasimetric_hidden_dim],
                w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                activation=jax.nn.relu,
                name="encoder",
            )

            x_sym, x_asym = jnp.split(encoder(state), 2, axis=-1)
            y_sym, y_asym = jnp.split(encoder(goal), 2, axis=-1)

            d_sym = jnp.sqrt(jnp.sum(jnp.square(x_sym - y_sym) + 1e-8, axis=-1))
            d_asym = jnp.max(jax.nn.relu(x_asym - y_asym), axis=-1)

            dist = d_sym + d_asym

        elif quasimetric == "max":
            encoder = hk.nets.MLP(
                list(hidden_layer_sizes) + [quasimetric_hidden_dim],
                w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                activation=jax.nn.relu,
                name="encoder",
            )

            x = encoder(state)
            y = encoder(goal)

            dist = jnp.max(jax.nn.relu(x - y), axis=-1)
        else:
            raise NotImplementedError

        if reentrant and twin_q:
            rng = hk.next_rng_key() if hk.running_init() else None
            twin_fn = hk.without_apply_rng(hk.transform(_quasimetric_fn))
            twin_params = hk.lift(twin_fn.init, name="inner")(
                rng, obs_, action_, goal_, dummy_action, reentrant=False
            )
            dist2 = twin_fn.apply(
                twin_params, obs_, action_, goal_, dummy_action, reentrant=False
            )
            dist = jnp.stack([dist, jnp.squeeze(dist2, axis=-1)], axis=-1)
        else:
            dist = dist[..., None]

        return dist

    def _critic_fn(obs, action, goal):
        if use_image_obs:
            obs = _unflatten_img(obs)
            goal = _unflatten_img(goal)
            img_encoder = TORSO()
            state = img_encoder(obs)
            goal = img_encoder(goal)
        else:
            state = obs
            goal = goal

        # negative softplus activation: quasimetric is non-negative
        critic_network = hk.Sequential(
            [
                hk.nets.MLP(
                    list(hidden_layer_sizes) + [1],
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                    activation=jax.nn.relu,
                    name="critic_network",
                ),
                # lambda x: -jax.nn.softplus(x),
            ]
        )
        q = critic_network(jnp.concatenate([state, action, goal], axis=-1))

        if twin_q:
            critic_network2 = hk.Sequential(
                [
                    hk.nets.MLP(
                        list(hidden_layer_sizes) + [1],
                        w_init=hk.initializers.VarianceScaling(
                            1.0, "fan_in", "uniform"
                        ),
                        activation=jax.nn.relu,
                        name="critic_network2",
                    ),
                    # lambda x: -jax.nn.softplus(x),
                ]
            )

            q2 = critic_network2(jnp.concatenate([state, action, goal], axis=-1))
            q = jnp.concatenate([q, q2], axis=-1)

        return q

    def _actor_fn(obs_and_goal):
        if use_image_obs:
            obs, goal = obs_and_goal[:, :obs_dim], obs_and_goal[:, obs_dim:]
            obs = _unflatten_img(obs)
            goal = _unflatten_img(goal)
            img_encoder = TORSO()
            state_and_goal = img_encoder(jnp.concatenate([obs, goal], axis=-1))
        else:
            state_and_goal = obs_and_goal
        network = hk.Sequential(
            [
                hk.nets.MLP(
                    list(hidden_layer_sizes),
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                    activation=jax.nn.relu,
                    activate_final=True,
                ),
                networks_lib.NormalTanhDistribution(
                    num_dimensions, min_scale=actor_min_std
                ),
            ]
        )

        return network(state_and_goal)

    def _potential_fn(goal):
        if use_image_obs:
            obs = _unflatten_img(obs)
            goal = _unflatten_img(goal)
            img_encoder = TORSO()
            goal = img_encoder(goal)

        potential_network = hk.Sequential(
            [
                hk.nets.MLP(
                    list(hidden_layer_sizes) + [1],
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                    activation=jax.nn.relu,
                    name="potential_network",
                ),
            ]
        )
        value = potential_network(goal)[..., None]
        if twin_q:
            potential_network = hk.Sequential(
                [
                    hk.nets.MLP(
                        list(hidden_layer_sizes) + [1],
                        w_init=hk.initializers.VarianceScaling(
                            1.0, "fan_in", "uniform"
                        ),
                        activation=jax.nn.relu,
                        name="potential_network2",
                    ),
                ]
            )
            value2 = potential_network(goal)[..., None]
            value = jnp.concatenate([value, value2], axis=-1)

        return value

    policy = hk.without_apply_rng(hk.transform(_actor_fn))
    contrastive_fn = hk.without_apply_rng(hk.transform(_contrastive_fn))
    quasimetric_fn = hk.without_apply_rng(hk.transform(_quasimetric_fn))
    critic_fn = hk.without_apply_rng(hk.transform(_critic_fn))
    potential_fn = hk.without_apply_rng(hk.transform(_potential_fn))

    # Create dummy observations and actions to create network parameters.
    dummy_obs = utils.add_batch_dim(utils.ones_like(spec.observations)[:obs_dim])
    dummy_action = utils.add_batch_dim(utils.ones_like(spec.actions))
    dummy_goal = utils.add_batch_dim(utils.ones_like(spec.observations)[:obs_dim])
    dummy_obs_and_goal = utils.add_batch_dim(utils.ones_like(spec.observations))

    return MetricDistillationNetworks(
        contrastive_network=networks_lib.FeedForwardNetwork(
            lambda key: contrastive_fn.init(key, dummy_obs, dummy_action, dummy_goal),
            contrastive_fn.apply,
        ),
        quasimetric_network=networks_lib.FeedForwardNetwork(
            lambda key: quasimetric_fn.init(
                key, dummy_obs, dummy_action, dummy_goal, dummy_action
            ),
            quasimetric_fn.apply,
        ),
        critic_network=networks_lib.FeedForwardNetwork(
            lambda key: critic_fn.init(key, dummy_obs, dummy_action, dummy_goal),
            critic_fn.apply,
        ),
        policy_network=networks_lib.FeedForwardNetwork(
            lambda key: policy.init(key, dummy_obs_and_goal), policy.apply
        ),
        potential_network=networks_lib.FeedForwardNetwork(
            lambda key: potential_fn.init(key, dummy_goal), potential_fn.apply
        ),
        log_prob=lambda params, actions: params.log_prob(actions),
        sample=lambda params, key: params.sample(seed=key),
        sample_eval=lambda params, key: params.mode(),
    )
