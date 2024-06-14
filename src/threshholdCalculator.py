import numpy as np

import constants


class ThresholdCalculator:
    def __init__(self, tuning_config):
        self.tuning_config = tuning_config

    def _calculate_threshold(
        self, sorted_values, looseness, default_threshold, min_jump, jump_delta
    ):
        """
        Calculate the threshold based on the sorted values, looseness, and minimum jump.
        """
        half_looseness = (looseness + 1) // 2
        limit = len(sorted_values) - half_looseness
        max_jump, threshold = min_jump, default_threshold
        for i in range(half_looseness, limit):
            jump = sorted_values[i + half_looseness] - sorted_values[i - half_looseness]
            if jump > max_jump:
                max_jump = jump
                threshold = sorted_values[i - half_looseness] + jump / 2
        return max_jump, threshold

    def get_global_threshold(self, original_values, looseness=1):
        """
        Calculate the global threshold based on the original values and looseness.
        """
        # Get the threshold parameters from the tuning configuration
        PAGE_TYPE_FOR_THRESHOLD, MIN_JUMP, JUMP_DELTA = map(
            self.tuning_config.threshold_params.get,
            ["PAGE_TYPE_FOR_THRESHOLD", "MIN_JUMP", "JUMP_DELTA"],
        )

        # Set the default global threshold based on the page type
        global_default_threshold = (
            constants.GLOBAL_PAGE_THRESHOLD_WHITE
            if PAGE_TYPE_FOR_THRESHOLD == "white"
            else constants.GLOBAL_PAGE_THRESHOLD_BLACK
        )

        # Sort the original values
        sorted_values = sorted(original_values)

        # Calculate the first and second thresholds
        max_jump1, threshold1 = self._calculate_threshold(
            sorted_values, looseness, global_default_threshold, MIN_JUMP, JUMP_DELTA
        )
        max_jump2, threshold2 = self._calculate_threshold(
            sorted_values, looseness, global_default_threshold, MIN_JUMP, JUMP_DELTA
        )

        # If the second threshold is larger and sufficiently different from the first, update the first threshold
        if max_jump2 > max_jump1 and abs(threshold1 - threshold2) > JUMP_DELTA:
            max_jump1, threshold1 = max_jump2, threshold2

        # Calculate the global threshold and the lower and upper bounds
        global_threshold, lower_bound, upper_bound = (
            threshold1,
            threshold1 - max_jump1 // 2,
            threshold1 + max_jump1 // 2,
        )

        return global_threshold, lower_bound, upper_bound

    def get_local_threshold(self, q_values, global_threshold, no_outliers):
        """
        This function calculates the local threshold based on the given q_values.
        It assumes that the background color is uniformly gray or white, but not alternating.
        There are two cases considered: 0 jump and 1 jump.
        """
        config = self.tuning_config
        threshold_params = config.threshold_params

        # Sort the q_values for further calculations
        q_values = sorted(q_values)

        # If there are less than 3 q_values, calculate the threshold based on the max and min gap
        if len(q_values) < 3:
            threshold = (
                global_threshold
                if max(q_values) - min(q_values) < threshold_params.MIN_GAP
                else np.mean(q_values)
            )
        else:
            # Initialize maximum jump and threshold
            max_jump, threshold = threshold_params.MIN_JUMP, 255

            # Calculate the maximum jump and corresponding threshold
            for i in range(1, len(q_values) - 1):
                current_jump = q_values[i + 1] - q_values[i - 1]
                if current_jump > max_jump:
                    max_jump = current_jump
                    threshold = q_values[i - 1] + current_jump / 2

            # Check if the maximum jump is less than the confident jump
            confident_jump = (
                threshold_params.MIN_JUMP + threshold_params.CONFIDENT_SURPLUS
            )
            if max_jump < confident_jump:
                if no_outliers:
                    threshold = global_threshold

        return threshold
