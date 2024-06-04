import os
# import glob
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import scipy.interpolate as interp
from itertools import cycle
from scipy.ndimage import gaussian_filter1d


def filter_outliers(x, y, m=100.0, eps=1e-6):
    """
    Reference: https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
    """
    # indices = np.all(
    #     abs(y - np.mean(y, axis=-1, keepdims=True)) < m * np.std(y, axis=-1, keepdims=True),
    #     axis=-1)
    # filtered_x, filtered_y = x[indices], y[indices]
    d = np.abs(y - np.median(y, axis=0, keepdims=True))
    mdev = np.median(d, axis=0, keepdims=True)
    s = d / (mdev + eps)
    indices = np.all(s < m, axis=-1)
    # data = y[..., 0]
    # d = np.abs(data - np.median(data))
    # mdev = np.median(d)
    # s = d / mdev if mdev else 0.
    # indices = s < m

    filtered_x, filtered_y = x[indices], y[indices]

    return filtered_x, filtered_y


def window_smooth(y, window_width=20, smooth_coef=0.05):
    window_size = int(window_width / 2)
    y_padding = np.concatenate([y[:1] for _ in range(window_size)], axis=0).flatten()
    y = np.concatenate([y_padding, y], axis=0)
    y_padding = np.concatenate([y[-1:] for _ in range(window_size)], axis=0).flatten()
    y = np.concatenate([y, y_padding], axis=0)

    coef = list()
    for i in range(window_width + 1):
        coef.append(np.exp(- smooth_coef * abs(i - window_size)))
    coef = np.array(coef)

    yw = list()
    for t in range(len(y) - window_width):
        yw.append(np.sum(y[t:t + window_width + 1] * coef) / np.sum(coef))

    return np.array(yw).flatten()


def collect_data(root_exp_log_dir, stats, algos,
                 timestep_field, max_steps):
    data = {}
    for module_subdir, stats_field, stats_name in stats:
        stats_data = {}
        for algo, algo_dir in algos:
            try:
                exp_dir = os.path.join(root_exp_log_dir, algo_dir)
                # algo_data = []
                seed_dirs = [os.path.join(exp_dir, exp_subdir)
                             for exp_subdir in os.listdir(exp_dir)
                             if re.search(r'^\d+$', exp_subdir)]

                csv_paths = [os.path.join(seed_dir, 'logs', module_subdir, 'logs.csv')
                             for seed_dir in seed_dirs]
                csv_paths = [csv_path for csv_path in csv_paths if os.path.exists(csv_path)]

                # df = pd.read_csv(csv_path)
                algo_data = []
                algo_data_timesteps = []
                algo_data_values = []
                for idx, csv_path in enumerate(csv_paths):
                    numlines = sum(1 for l in open(csv_path))
                    if numlines < 2: # header line
                        print(f"CSV path {csv_path} has less than 2 lines")
                        continue
                    df = pd.read_csv(csv_path)

                    try:
                        df = df.drop_duplicates(timestep_field, keep='last')
                        df = df[df[timestep_field] <= max_steps]
                        values = df[stats_field].values
                    except:
                        print(f"CSV path {csv_path} has no stats field {stats_field}")
                        continue
                    timesteps = df[timestep_field].values
                    invalid_mask = np.isinf(values) | np.isnan(values)
                    if np.any(invalid_mask):
                        values = values[~invalid_mask]
                        timesteps = timesteps[~invalid_mask]

                    algo_data_timesteps.append(timesteps)
                    algo_data_values.append(values)

                if algo_data_timesteps:
                    # Interpolate to the max timesteps
                    ref_idx, max_timestep = 0, 0
                    ref_timesteps = None
                    for idx, timesteps in enumerate(algo_data_timesteps):
                        if timesteps[-1] >= max_timestep:
                            ref_idx = idx
                            ref_timesteps = timesteps
                            max_timestep = timesteps[-1]

                    algo_data.append(ref_timesteps)

                    for idx, (timesteps, values) in enumerate(zip(
                            algo_data_timesteps, algo_data_values)):
                        if idx != ref_idx:
                            interpolation = interp.interp1d(
                                timesteps, values,
                                bounds_error=False,
                                fill_value=(values[0], values[-1]))
                            interp_values = interpolation(ref_timesteps)
                            algo_data.append(interp_values)
                        else:
                            algo_data.append(values)
                    algo_data = np.asarray(algo_data).T

                # df = pd.concat((pd.read_csv(f) for f in all_csv_paths), ignore_index=True)
            except FileNotFoundError:
                print(f"CSV path not found: {csv_path}")
                continue

            if isinstance(algo_data, np.ndarray):
                stats_data[algo] = algo_data
        data[stats_field] = stats_data

    return data


def main(args):
    root_exp_log_dir = os.path.expanduser(args.root_exp_log_dir)
    assert os.path.exists(root_exp_log_dir), \
        "Cannot find root_data_exp_dir: {}".format(root_exp_log_dir)
    fig_save_dir = os.path.expanduser(args.fig_save_dir)
    os.makedirs(fig_save_dir, exist_ok=True)

    nrows = np.ceil(len(args.stats) / 4).astype(int) + 1
    ncols = 4
    f, axes = plt.subplots(nrows=nrows, ncols=ncols)
    if nrows == 1:
        axes = np.array([axes])
    f.set_figheight(8 * nrows)
    f.set_figwidth(8 * ncols)

    for ax in axes.flatten():
        ax.set_axis_off()

    # read all data
    data = collect_data(root_exp_log_dir, args.stats, args.algos,
                        args.timestep_field, args.max_steps)

    # plot
    num_curves = len(args.algos)
    cmap = plt.cm.get_cmap('tab20', num_curves)
    cycol = cycle(cmap.colors)
    for algo_idx, (algo, _) in enumerate(args.algos):
        c = next(cycol)
        for stat_idx, (_, stats_field, stats_name) in enumerate(args.stats):
            try:
                x = data[stats_field][algo][..., 0]
                y = data[stats_field][algo][..., 1:]


                # if stats_field in ['actor_loss',
                #                    'behavioral_cloning_loss',
                #                    'q_ratio',
                #                    'q_pos_ratio',
                #                    'q_neg_ratio']:
                #     x, y = filter_outliers(x, y)

                # mean = window_smooth(np.mean(y, axis=-1))
                # std = window_smooth(np.std(y, axis=-1))
                # we used success_rate_1000 and don't need smoothing
                mean = np.mean(y, axis=-1)
                std = np.std(y, axis=-1) / np.sqrt(y.shape[-1])
                # if stats_field == 'success_1000' and 'md (max)' in algo:
                #     print()
                nseeds = y.shape[-1]
                ax = axes[stat_idx // 4, stat_idx % 4]
                ax.set_axis_on()

                ax.plot(x, mean, label=algo + f' ({nseeds} seeds)', color=c)
                

                smoothed_mean = mean#(mean[max(0, len(x) // 2):])
                smoothed_std = std#(std[max(0, len(x) // 2):])
                #ax.set_ylim([np.min(smoothed_mean - np.sqrt(nseeds) * smoothed_std), np.max(smoothed_mean + smoothed_std * np.sqrt(nseeds))])
                if stats_field in args.log_scale_stats:
                    ax.set_yscale('log')
                if stats_field in args.symlog_scale_stats:
                    ax.set_yscale('symlog')
                ax.fill_between(x, mean - std, mean + std,
                                facecolor=c, alpha=0.35)
                ax.set_xlabel(args.timestep_field.replace('_', ' '),
                              fontsize=14)
                ax.set_ylabel(stats_name, fontsize=14)
                ax.ticklabel_format(style='sci', scilimits=(-3, 4), axis='x')
            except KeyError:
                print("[Warning] Algorithm {} data not found".format(algo))
    lines, labels = axes[0, 0].get_legend_handles_labels()

    f.legend(lines, labels, fontsize=18, loc='lower center', bbox_to_anchor=(0.5, 0.1))
    f_path = os.path.abspath(os.path.join(fig_save_dir, args.fig_filename + '.pdf'))
    f.suptitle(args.fig_title.replace('_', ' ').capitalize(), fontsize=36, y=1.)
    plt.tight_layout(pad=1.2)
    plt.savefig(fname=f_path)
    print(f"Save figure to: {f_path}")

    # # if args.show_xlabel:
    # #     axes[stat_idx].set_xlabel('gradient steps', fontsize=14)
    # # axes[stat_idx].set_xlim([-5 * int(stats_x_scale), 305 * int(stats_x_scale)])
    # axes[stat_idx].ticklabel_format(style='sci', scilimits=(-3, 4), axis='x')
    # # axes[stat_idx].yaxis.set_tick_params(labelsize=16)
    # axes[stat_idx].set_yticks([0.0, 0.5, 1.0])
    # # axes[stat_idx].set_ylim([-0.05, 1.05])
    # axes[stat_idx].yaxis.set_minor_locator(MultipleLocator(0.1))
    # axes[stat_idx].set_ylabel(stats_name, fontsize=14)
    # axes[stat_idx].grid(b=True)


if __name__ == "__main__":
    # custom argument type
    def str_pair(s):
        splited_s = s.split(',')
        assert splited_s, 'invalid string pair'
        return splited_s[0], splited_s[1]

    def str_triplet(s):
        splited_s = s.split(',')
        assert splited_s, 'invalid string pair'
        return splited_s[0], splited_s[1], splited_s[2]

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_exp_log_dir', type=str,
                        default='/projects/rsalakhugroup/chongyiz/td_cpc_logs/')
    parser.add_argument('--fig_title', type=str, default='Fetch Reach')
    parser.add_argument('--fig_save_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'figures'))
    parser.add_argument('--fig_filename', type=str,
                        default='fetch_reach')
    parser.add_argument('--algos', type=str_pair, nargs='+', default=[
        ('Metric Distillation', '20240127_metric_distillation_fetch_reach'),
    ])
    parser.add_argument('--stats', type=str_triplet, nargs='+', default=[
        ('evaluator', 'success', 'Success Rate'),
        ('evaluator', 'success_10', 'Success Rate 10'),
        ('evaluator', 'success_100', 'Success Rat 100'),
        ('evaluator', 'success_1000', 'Success Rate 1000'),
        ('evaluator', 'final_dist', 'Final Distance'),
        ('learner', 'actor_loss', 'Actor Loss'),
        ('learner', 'critic_loss', 'Critic Loss'),
    ])
    parser.add_argument('--log_scale_stats', type=str, nargs='+', default=[])
    parser.add_argument('--symlog_scale_stats', type=str, nargs='+', default=[])
    parser.add_argument('--timestep_field', type=str, default='actor_steps')
    parser.add_argument('--max_steps', type=int, default=np.iinfo(np.int64).max)
    args = parser.parse_args()

    main(args)
