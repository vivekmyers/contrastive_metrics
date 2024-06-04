# coding=utf-8
# Copyright 2023 The Google Research Authors.
# Copyright 2023 Chongyi Zheng.
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

# python3

import os
import functools
import dataclasses
import json
import tensorflow as tf
import launchpad as lp
from absl import app
from absl import flags
import logging
from metric_distillation.agents import DistributedMetricDistillation
from metric_distillation.networks import make_networks
from metric_distillation.utils import make_environment, goal_to_obs_2d
from metric_distillation.config import MetricDistillationConfig


r"""Example running Contrastive Metric Distillation in JAX.

Run using multi-processing (required for image-based experiments):
  python run.py --lp_launch_type=local_mp

Run using multi-threading
  python run.py --lp_launch_type=local_mt
"""



# disable tensorflow_probability warning: The use of `check_types` is deprecated and does not have any effect.
logger = logging.getLogger("root")


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())


FLAGS = flags.FLAGS
flags.DEFINE_boolean("debug", False, "Runs training for just a few steps.")
flags.DEFINE_bool("run_tf_eagerly", False, "Enables / disables eager execution of tf.functions.")
flags.DEFINE_string(
    "exp_log_dir",
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "metric_distillation_logs"),
    "Root directory for writing logs/summaries/checkpoints.",
)
flags.DEFINE_bool("exp_log_dir_add_uid", False, "Enables / disables unique id for the log directory")
flags.DEFINE_string("env_name", "offlin_ant_umaze", "Select an environment")
flags.DEFINE_integer(
    "max_number_of_steps",
    500_000,
    "For online RL experiments, max_number_of_steps is the number of "
    "environment steps. For offline RL experiments, this is the number of"
    "gradient steps.",
)
flags.DEFINE_bool("jit", True, "Enables / disables jax.jit compilation")
flags.DEFINE_integer("seed", 0, "Random seed")

flags.DEFINE_float("discount", 0.99, "Discount for TD updates.")
flags.DEFINE_float("tau", 0.005, "Target smoothing coefficient.")
flags.DEFINE_integer("batch_size", 256, "Batch size for updates.")
flags.DEFINE_float("critic_learning_rate", 5e-5, "Learning rate.")
flags.DEFINE_float("actor_learning_rate", 5e-5, "Learning rate.")
flags.DEFINE_float("contrastive_learning_rate", 1e-5, "Learning rate.")
flags.DEFINE_float("quasimetric_learning_rate", 5e-5, "Learning rate.")

flags.DEFINE_float("bc_coef", 0.2, "Behavior cloning coefficient.")
flags.DEFINE_string("bc_loss", "mse", "Behavior cloning loss.")
flags.DEFINE_integer("repr_dim", 64, "Size of representation.")
flags.DEFINE_boolean("repr_norm", True, "Whether to normalize the representation.")
flags.DEFINE_float("repr_norm_temp", 0.1, "Temperature for representation normalization.")
flags.DEFINE_boolean(
    "adaptive_repr_norm_temp", True, "Whether to use adaptive temperature for representation normalization."
)

flags.DEFINE_boolean("twin_q", False, "Whether to use twin Q networks.")
flags.DEFINE_boolean("use_gcbc", False, "Whether to use GCBC.")
flags.DEFINE_boolean("use_cpc", False, "Whether to use Contrastive CPC.")
flags.DEFINE_boolean("use_nce", False, "Whether to use Contrastive NCE.")
flags.DEFINE_boolean("onestep", False, "Whether to jointly train the quasimetric in one step.")
flags.DEFINE_boolean("nopotential", False, "Whether to use goal potential c(g) in CMD-1")

flags.DEFINE_float("random_goals", 0.0, "Probability of using random goals.")
flags.DEFINE_float("lam_init", 1.0, "Initial value of lambda.")
flags.DEFINE_float("lam_scale", 1.0, "Scale of lambda.")
flags.DEFINE_float("margin", 0.35, "Margin for the constraint.")

flags.DEFINE_string("quasimetric", "mrn", "Quasimetric parameterization")
flags.DEFINE_integer("quasimetric_hidden_dim", 256, "Hidden dimension for quasimetric.")
flags.DEFINE_integer("quasimetric_num_groups", 32, "Number of groups for quasimetric.")
flags.DEFINE_float("quasimetric_action_constraint_coef", 1e-5, "Coefficient for fixed dual actor loss.")
flags.DEFINE_boolean("contrastive_only", False, "Whether to use contrastive triangle inequality.")
flags.DEFINE_float("triangle_ineq_coef", 0.0, "Coefficient for triangle inequality.")
flags.DEFINE_boolean("awr", False, "Whether to use AWAC.")
flags.DEFINE_float("awr_temp", 1.0, "Temperature for AWAC.")

flags.DEFINE_float("mixup_alpha", 2.0, "Parameter of the Dirichlet prior to sample the mixing coefficient.")
flags.DEFINE_float("mixup_bandwidth", 2.75, "Bandwidth for computing the mix-up probability.")


@functools.lru_cache()
def get_env(env_name, start_index, end_index):
    return make_environment(env_name, start_index, end_index, seed=0)


def get_program(params):
    """Constructs the program."""

    env_name = params["env_name"]
    seed = params.pop("seed")

    if params.get("use_image_obs", False) and not params.get("local", False):
        print("WARNING: overwriting parameters for image-based tasks.")
        params["num_sgd_steps_per_step"] = 8
        params["prefetch_size"] = 8
        params["num_actors"] = 5

    if env_name.startswith("offline"):
        # No actors needed for the offline RL experiments. Evaluation is
        # handled separately.
        params["num_actors"] = 0

    config = MetricDistillationConfig(**params)

    env_factory = lambda seed: make_environment(env_name, config.start_index, config.end_index, seed)

    env_factory_no_extra = lambda seed: env_factory(seed)[0]  # Remove obs_dim.
    environment, obs_dim = get_env(env_name, config.start_index, config.end_index)
    assert (environment.action_spec().minimum == -1).all()
    assert (environment.action_spec().maximum == 1).all()
    config.obs_dim = obs_dim
    config.max_episode_steps = getattr(environment, "_step_limit")
    if env_name == "offline_ant_umaze_diverse":
        # This environment terminates after 700 steps, but demos have 1000 steps.
        config.max_episode_steps = 1000

    network_factory = functools.partial(
        make_networks,
        obs_dim=obs_dim,
        goal_to_obs=functools.partial(
            goal_to_obs_2d, obs_dim=config.obs_dim, start_index=config.start_index, end_index=config.end_index
        ),
        repr_dim=config.repr_dim,
        repr_norm=config.repr_norm,
        repr_norm_temp=config.repr_norm_temp,
        adaptive_repr_norm_temp=config.adaptive_repr_norm_temp,
        twin_q=config.twin_q,
        use_image_obs=config.use_image_obs,
        hidden_layer_sizes=config.hidden_layer_sizes,
        quasimetric=config.quasimetric,
        quasimetric_hidden_dim=config.quasimetric_hidden_dim,
        quasimetric_num_groups=config.quasimetric_num_groups,
    )

    agent = DistributedMetricDistillation(
        seed=seed,
        environment_factory=env_factory_no_extra,
        network_factory=network_factory,
        config=config,
        num_actors=config.num_actors,
        log_to_bigtable=True,
        max_number_of_steps=config.max_number_of_steps,
        log_dir=FLAGS.exp_log_dir,
        log_dir_add_uid=FLAGS.exp_log_dir_add_uid,
    )

    # save config to log_dir
    with open(os.path.join(FLAGS.exp_log_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(config), f)

    return agent.build()


def main(_):
    if FLAGS.run_tf_eagerly:
        tf.config.run_functions_eagerly(True)  # debug tensorflow functions

    # 1. Select an environment.
    # Supported environments:
    #   OpenAI Gym Fetch: fetch_{reach,push,pick_and_place,slide}
    # Image observation environments:
    #   OpenAI Gym Fetch: fetch_{reach,push,pick_and_place,slide}_image
    # Offline environments:
    #   antmaze: offline_ant_{umaze,umaze_diverse,
    #                         medium_play,medium_diverse,
    #                         large_play,large_diverse}
    env_name = FLAGS.env_name
    params = {
        "seed": FLAGS.seed,
        "jit": FLAGS.jit,
        "use_random_actor": True,
        "env_name": env_name,
        # For online RL experiments, max_number_of_steps is the number of
        # environment steps. For offline RL experiments, this is the number of
        # gradient steps.
        "max_number_of_steps": FLAGS.max_number_of_steps,
        "use_image_obs": "image" in env_name,
        "discount": FLAGS.discount,
        "tau": FLAGS.tau,
        "batch_size": FLAGS.batch_size,
        "critic_learning_rate": FLAGS.critic_learning_rate,
        "actor_learning_rate": FLAGS.actor_learning_rate,
        "contrastive_learning_rate": FLAGS.contrastive_learning_rate,
        "quasimetric_learning_rate": FLAGS.quasimetric_learning_rate,
        "bc_coef": FLAGS.bc_coef,
        "repr_dim": FLAGS.repr_dim,
        "repr_norm": FLAGS.repr_norm,
        "repr_norm_temp": FLAGS.repr_norm_temp,
        "adaptive_repr_norm_temp": FLAGS.adaptive_repr_norm_temp,
        "twin_q": FLAGS.twin_q,
        "use_gcbc": FLAGS.use_gcbc,
        "random_goals": FLAGS.random_goals,
        "lam_init": FLAGS.lam_init,
        "lam_scale": FLAGS.lam_scale,
        "margin": FLAGS.margin,
        "quasimetric": FLAGS.quasimetric,
        "quasimetric_hidden_dim": FLAGS.quasimetric_hidden_dim,
        "quasimetric_num_groups": FLAGS.quasimetric_num_groups,
        "contrastive_only": FLAGS.contrastive_only,
        "quasimetric_action_constraint_coef": FLAGS.quasimetric_action_constraint_coef,
        "awr": FLAGS.awr,
        "awr_temp": FLAGS.awr_temp,
        "onestep": FLAGS.onestep,
        "mixup_alpha": FLAGS.mixup_alpha,
        "mixup_bandwidth": FLAGS.mixup_bandwidth,
    }
    if "ant_" in env_name or "maze2d_" in env_name or "point_" in env_name:
        params["end_index"] = 2

    # For the offline RL experiments, modify some hyperparameters.
    if env_name.startswith("offline"):
        params.update(
            {
                # Effectively remove the rate-limiter by using very large values.
                "samples_per_insert": 1_000_000,
                "samples_per_insert_tolerance_rate": 100_000_000.0,
                # For the actor update, only use future states as goals.
                "random_goals": FLAGS.random_goals,
                "twin_q": FLAGS.twin_q,  # Learn two critics, and take the minimum.
                # 'repr_dim': 16,  # Decrease the representation size 64 --> 16.
                "bc_coef": FLAGS.bc_coef,  # Add a behavioral cloning term to the actor. AntMazeLarge MSE loss.
                "bc_loss": FLAGS.bc_loss,
                "batch_size": 1024,  # Increase the batch size 256 --> 1024.
                # Increase the policy network size (512, 512, 512, 512) --> (1024, 1024, 1024, 1024)
                "hidden_layer_sizes": (1024, 1024, 1024, 1024),  # AntMaze doesn't seem to need a large network,
            }
        )

    # 2. Select compute parameters. The default parameters are already tuned, so
    # use this mainly for debugging.
    if FLAGS.debug:
        params.update(
            {
                "min_replay_size": 4_000,
                "local": True,
                "num_sgd_steps_per_step": 1,
                "prefetch_size": 1,
                "num_actors": 1,
                "batch_size": 32,
                "max_number_of_steps": 20_000,
                "hidden_layer_sizes": (32, 32),
            }
        )

    program = get_program(params)
    # Set terminal='tmux' if you want different components in different windows.
    lp.launch(program, terminal="current_terminal")


if __name__ == "__main__":
    app.run(main)
