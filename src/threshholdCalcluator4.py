import numpy as np
import matplotlib.pyplot as plt
from constants import GLOBAL_PAGE_THRESHOLD_WHITE, GLOBAL_PAGE_THRESHOLD_BLACK

import constants
from defaults.config import CONFIG_DEFAULTS


class ThresholdCalculator:

    def __init__(self):
        self.tuning_config = CONFIG_DEFAULTS
        self.constants = constants

    def get_global_threshold(
        self,
        q_vals_orig,
        plot_title=None,
        plot_show=True,
        sort_in_plot=True,
        looseness=1,
    ):
        """
        Calculate global threshold.

        Args:
            q_vals_orig (list): List of values.
            plot_title (str): Title for the plot.
            plot_show (bool): Whether to show the plot.
            sort_in_plot (bool): Whether to sort values in the plot.
            looseness (int): Looseness factor.

        Returns:
            tuple: Global threshold, threshold low, threshold high.
        """

        PAGE_TYPE_FOR_THRESHOLD = (
            "black"
            if self.tuning_config.threshold_params.get("PAGE_TYPE_FOR_THRESHOLD")
            == "black"
            else "white"
        )
        MIN_JUMP = self.tuning_config.threshold_params.get("MIN_JUMP")
        JUMP_DELTA = self.tuning_config.threshold_params.get("JUMP_DELTA")

        global_default_threshold = (
            GLOBAL_PAGE_THRESHOLD_WHITE
            if PAGE_TYPE_FOR_THRESHOLD == "white"
            else GLOBAL_PAGE_THRESHOLD_BLACK
        )

        q_vals = sorted(q_vals_orig)
        ls = (looseness + 1) // 2
        l = len(q_vals) - ls
        max1, thr1 = MIN_JUMP, global_default_threshold

        for i in range(ls, l):
            jump = q_vals[i + ls] - q_vals[i - ls]
            if jump > max1:
                max1 = jump
                thr1 = q_vals[i - ls] + jump / 2

        max2, thr2 = MIN_JUMP, global_default_threshold

        for i in range(ls, l):
            jump = q_vals[i + ls] - q_vals[i - ls]
            new_thr = q_vals[i - ls] + jump / 2
            if jump > max2 and abs(thr1 - new_thr) > JUMP_DELTA:
                max2 = jump
                thr2 = new_thr

        global_thr = thr1
        j_low, j_high = thr1 - max1 // 2, thr1 + max1 // 2

        if plot_title:
            _, ax = plt.subplots()
            ax.bar(range(len(q_vals_orig)), q_vals if sort_in_plot else q_vals_orig)
            ax.set_title(plot_title)
            thrline = ax.axhline(global_thr, color="green", ls="–", linewidth=5)
            thrline.set_label("Global Threshold")
            thrline = ax.axhline(thr2, color="red", ls=":", linewidth=3)
            thrline.set_label("THR2 Line")

            ax.set_ylabel("Values")
            ax.set_xlabel("Position")
            ax.legend()
            if plot_show:
                plt.title(plot_title)
                plt.show()

        return global_thr, j_low, j_high

    def get_local_threshold(
        self, q_vals, global_thr, no_outliers, plot_title=None, plot_show=True
    ):
        """
        Calculate local threshold.

        Args:
            q_vals (list): List of values.
            global_thr (float): Global threshold.
            no_outliers (bool): Flag to consider outliers.
            plot_title (str): Title for the plot.
            plot_show (bool): Whether to show the plot.

        Returns:
            float: Local threshold.
        """

        if len(q_vals) < 3:
            thr1 = (
                global_thr
                if np.max(q_vals) - np.min(q_vals)
                < self.tuning_config.threshold_params.MIN_GAP
                else np.mean(q_vals)
            )
        else:
            l = len(q_vals) - 1
            max1, thr1 = self.tuning_config.threshold_params.MIN_JUMP, 255
            for i in range(1, l):
                jump = q_vals[i] - q_vals[i - 1]
                if jump > max1:
                    max1 = jump
                    thr1 = q_vals[i - 1] + jump / 2

            confident_jump = (
                self.tuning_config.threshold_params.MIN_JUMP
                + self.tuning_config.threshold_params.CONFIDENT_SURPLUS
            )

            if max1 < confident_jump:
                if no_outliers:
                    thr1 = global_thr

        if plot_show and plot_title is not None:
            _, ax = plt.subplots()
            ax.bar(range(len(q_vals)), q_vals)
            thrline = ax.axhline(thr1, color="green", ls=("-."), linewidth=3)
            thrline.set_label("Local Threshold")
            thrline = ax.axhline(global_thr, color="red", ls=":", linewidth=5)
            thrline.set_label("Global Threshold")
            ax.set_title(plot_title)
            ax.set_ylabel("Bubble Mean Intensity")
            ax.set_xlabel("Bubble Number(sorted)")
            ax.legend()

            if plot_show:
                plt.show()

        return thr1
