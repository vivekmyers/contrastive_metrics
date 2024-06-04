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

"""Metric Distillation config."""
import dataclasses
from typing import Optional, Union, Tuple

from acme.adders import reverb as adders_reverb


@dataclasses.dataclass
class MetricDistillationConfig:
    """Configuration options for Metric Distillation."""

    env_name: str = ''
    max_number_of_steps: int = 1_000_000
    num_actors: int = 4

    # Loss options
    batch_size: int = 256
    contrastive_learning_rate: float = 3e-4
    quasimetric_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    discount: float = 0.99
    tau: float = 0.005  # target smoothing coefficient.
    hidden_layer_sizes: Tuple[int, ...] = (512, 512, 512, 512)

    # Replay options
    min_replay_size: int = 10000
    max_replay_size: int = 1000000
    replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
    prefetch_size: int = 4
    num_parallel_calls: Optional[int] = 4
    samples_per_insert: float = 256
    # Rate to be used for the SampleToInsertRatio rate limitter tolerance.
    # See a formula in make_replay_tables for more details.
    samples_per_insert_tolerance_rate: float = 0.1
    num_sgd_steps_per_step: int = 64  # Gradient updates to perform per step.

    repr_dim: Union[int, str] = 64  # Size of representation.
    use_random_actor: bool = True  # Initial with uniform random policy.
    repr_norm: bool = True
    repr_norm_temp: float = 0.01
    adaptive_repr_norm_temp: bool = True
    local: bool = False  # Whether running locally. Disables eval.
    twin_q: bool = False
    use_cpc: bool = True
    use_nce: bool = False
    use_gcbc: bool = False
    use_image_obs: bool = False
    random_goals: float = 1.0
    jit: bool = True
    bc_coef: float = 0.0
    bc_loss: str = 'mse'
    lam_init: float = 1.0
    lam_scale: float = 1.0
    margin: float = 0.35  # margin for the constraint
    quasimetric: str = 'max'  # IQE, MRN, MAX
    quasimetric_hidden_dim: int = 256  # 64 for iqe, 256 for max / mrn
    quasimetric_num_groups: int = 32  # 8 for iqe, 32 for max / mrn
    dual_actor_loss: bool = False  # DELETEME (chongyi)
    fixed_dual_actor_loss: float = 0.0  # DELETEME (chongyi)
    contrastive_only: bool = False
    triangle_ineq_coef: float = 0.0
    awr: bool = False
    awr_temp: float = 1.0
    onestep: bool = False
    nopotential: bool = False
    mixup_alpha: float = 2.0
    mixup_bandwidth: float = 2.75

    # Parameters that should be overwritten, based on each environment.
    obs_dim: int = -1
    max_episode_steps: int = -1
    start_index: int = 0
    end_index: int = -1
