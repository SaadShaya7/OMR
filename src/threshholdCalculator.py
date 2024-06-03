import matplotlib.pyplot as plt
from typing import Tuple, List
import numpy as np

from src import constants
from defaults.config import CONFIG_DEFAULTS


class ThresholdCalculator:
    def __init__(self):
        self.tuning_config = CONFIG_DEFAULTS
        self.constants = constants

    def get_global_threshold(
        self,
        q_values: List[float],
        plot_title: str = None,
        show_plot: bool = True,
        sort_values: bool = True,
        looseness_factor: int = 1,
    ) -> Tuple[float, float, float]:
        """
        Calculate the global threshold based on q_values.
        There can be either 1 or 2 significant jumps in the values indicating thresholds.
        The method finds the first large gap which is considered the safe threshold.
        """
        page_type = self.tuning_config.threshold_params.get("PAGE_TYPE_FOR_THRESHOLD")
        min_jump = self.tuning_config.threshold_params.get("MIN_JUMP")
        jump_delta = self.tuning_config.threshold_params.get("JUMP_DELTA")

        default_threshold = (
            self.constants.GLOBAL_PAGE_THRESHOLD_WHITE
            if page_type == "white"
            else self.constants.GLOBAL_PAGE_THRESHOLD_BLACK
        )

        sorted_q_values = sorted(q_values)
        half_looseness = (looseness_factor + 1) // 2
        threshold_length = len(sorted_q_values) - half_looseness

        max_jump1, threshold1 = min_jump, default_threshold
        for i in range(half_looseness, threshold_length):
            current_jump = (
                sorted_q_values[i + half_looseness]
                - sorted_q_values[i - half_looseness]
            )
            if current_jump > max_jump1:
                max_jump1 = current_jump
                threshold1 = sorted_q_values[i - half_looseness] + current_jump / 2

        max_jump2, threshold2 = min_jump, default_threshold
        for i in range(half_looseness, threshold_length):
            current_jump = (
                sorted_q_values[i + half_looseness]
                - sorted_q_values[i - half_looseness]
            )
            new_threshold = sorted_q_values[i - half_looseness] + current_jump / 2
            if (
                current_jump > max_jump2
                and abs(threshold1 - new_threshold) > jump_delta
            ):
                max_jump2 = current_jump
                threshold2 = new_threshold

        threshold = threshold1
        lower_bound = threshold - max_jump1 / 2
        upper_bound = threshold + max_jump1 / 2

        if plot_title:
            self._plot_thresholds(
                q_values,
                sorted_q_values,
                threshold,
                plot_title,
                show_plot,
                sort_values,
                threshold2,
            )

        return threshold, lower_bound, upper_bound

    def get_local_threshold(
        self,
        q_values: List[float],
        global_threshold: float,
        no_outliers: bool,
        plot_title: str = None,
        show_plot: bool = True,
    ) -> float:
        """
        Calculate the local threshold based on q_values and the global threshold.
        This method considers the case of 0 or 1 significant jumps in the values.
        """
        sorted_q_values = sorted(q_values)
        min_gap = self.tuning_config.threshold_params.get("MIN_GAP")
        min_jump = self.tuning_config.threshold_params.get("MIN_JUMP")
        confident_surplus = self.tuning_config.threshold_params.get("CONFIDENT_SURPLUS")

        if len(sorted_q_values) < 3:
            local_threshold = (
                global_threshold
                if np.max(sorted_q_values) - np.min(sorted_q_values) < min_gap
                else np.mean(sorted_q_values)
            )
        else:
            max_jump, local_threshold = min_jump, 255
            for i in range(1, len(sorted_q_values) - 1):
                jump = sorted_q_values[i + 1] - sorted_q_values[i - 1]
                if jump > max_jump:
                    max_jump = jump
                    local_threshold = sorted_q_values[i - 1] + jump / 2

            confident_jump = min_jump + confident_surplus
            if max_jump < confident_jump and not no_outliers:
                local_threshold = global_threshold

        if show_plot and plot_title is not None:
            self._plot_thresholds(
                q_values,
                sorted_q_values,
                local_threshold,
                plot_title,
                show_plot,
                False,
                global_threshold,
            )

        return local_threshold

    def _plot_thresholds(
        self,
        original_values,
        sorted_values,
        threshold,
        title,
        show,
        sort_in_plot,
        additional_threshold=None,
    ):
        _, ax = plt.subplots()
        ax.bar(
            range(len(original_values)),
            sorted_values if sort_in_plot else original_values,
        )
        ax.set_title(title)
        ax.axhline(
            threshold,
            color="green",
            linestyle="--",
            linewidth=5,
            label="Global Threshold",
        )
        ax.set_ylabel("Values")
        ax.set_xlabel("Position")
        ax.legend()

        if additional_threshold is not None:
            ax.axhline(
                additional_threshold,
                color="red",
                linestyle=":",
                linewidth=5,
                label="THR2 Line",
            )

        if show:
            plt.show()
