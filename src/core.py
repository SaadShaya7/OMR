import os
from collections import defaultdict
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

import src.constants as constants
from src.logger import logger
from src.utils.image import CLAHE_HELPER, ImageUtils


class ImageInstanceOps:
    """Class to hold fine-tuned utilities for a group of images. One instance for each processing directory."""

    save_img_list: Any = defaultdict(list)

    def __init__(self, tuning_config):
        super().__init__()
        self.tuning_config = tuning_config
        self.save_image_level = tuning_config.outputs.save_image_level

    def apply_preprocessors(self, file_path, in_omr, template):
        tuning_config = self.tuning_config

        in_omr = ImageUtils.resize_util(
            in_omr,
            tuning_config.dimensions.processing_width,
            tuning_config.dimensions.processing_height,
        )

        for pre_processor in template.pre_processors:
            in_omr = pre_processor.apply_filter(in_omr, file_path)
        return in_omr

    def read_omr_response(self, template, image, name, save_dir=None):
        config = self.tuning_config
        auto_align = config.alignment_params.auto_align
        try:
            img = image.copy()

            img = cv2.flip(img, 1)
            img = ImageUtils.resize_util(
                img, template.page_dimensions[0], template.page_dimensions[1]
            )
            if img.max() > img.min():
                img = ImageUtils.normalize_util(img)

            transp_layer = img.copy()
            final_marked = img.copy()

            morph = img.copy()
            self.append_save_img(3, morph)

            if auto_align:

                morph = CLAHE_HELPER.apply(morph)
                self.append_save_img(3, morph)

                morph = ImageUtils.adjust_gamma(
                    morph, config.threshold_params.GAMMA_LOW
                )

                _, morph = cv2.threshold(morph, 220, 220, cv2.THRESH_TRUNC)
                morph = ImageUtils.normalize_util(morph)
                self.append_save_img(3, morph)

            alpha = 0.65
            omr_response = {}
            multi_marked, multi_roll = 0, 0

            if auto_align:

                v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
                morph_v = cv2.morphologyEx(
                    morph, cv2.MORPH_OPEN, v_kernel, iterations=3
                )
                _, morph_v = cv2.threshold(morph_v, 200, 200, cv2.THRESH_TRUNC)
                morph_v = 255 - ImageUtils.normalize_util(morph_v)

                self.append_save_img(3, morph_v)

                morph_thr = 60
                _, morph_v = cv2.threshold(morph_v, morph_thr, 255, cv2.THRESH_BINARY)

                morph_v = cv2.erode(morph_v, np.ones((5, 5), np.uint8), iterations=2)

                self.append_save_img(3, morph_v)

                self.append_save_img(6, morph_v)

                for field_block in template.field_blocks:
                    s, d = field_block.origin, field_block.dimensions

                    match_col, max_steps, align_stride, thk = map(
                        config.alignment_params.get,
                        [
                            "match_col",
                            "max_steps",
                            "stride",
                            "thickness",
                        ],
                    )
                    shift, steps = 0, 0
                    while steps < max_steps:
                        left_mean = np.mean(
                            morph_v[
                                s[1] : s[1] + d[1],
                                s[0] + shift - thk : -thk + s[0] + shift + match_col,
                            ]
                        )
                        right_mean = np.mean(
                            morph_v[
                                s[1] : s[1] + d[1],
                                s[0]
                                + shift
                                - match_col
                                + d[0]
                                + thk : thk
                                + s[0]
                                + shift
                                + d[0],
                            ]
                        )

                        left_shift, right_shift = left_mean > 100, right_mean > 100
                        if left_shift:
                            if right_shift:
                                break
                            else:
                                shift -= align_stride
                        else:
                            if right_shift:
                                shift += align_stride
                            else:
                                break
                        steps += 1

                    field_block.shift = shift

            self.append_save_img(5, img)

            all_q_vals, all_q_strip_arrs, all_q_std_vals = [], [], []
            total_q_strip_no = 0
            for field_block in template.field_blocks:
                box_w, box_h = field_block.bubble_dimensions
                q_std_vals = []
                for field_block_bubbles in field_block.traverse_bubbles:
                    q_strip_vals = []
                    for pt in field_block_bubbles:

                        x, y = (pt.x + field_block.shift, pt.y)
                        rect = [y, y + box_h, x, x + box_w]
                        q_strip_vals.append(
                            cv2.mean(img[rect[0] : rect[1], rect[2] : rect[3]])[0]
                        )
                    q_std_vals.append(round(np.std(q_strip_vals), 2))
                    all_q_strip_arrs.append(q_strip_vals)

                    all_q_vals.extend(q_strip_vals)

                    total_q_strip_no += 1
                all_q_std_vals.extend(q_std_vals)

            global_std_thresh, _, _ = self.get_global_threshold(all_q_std_vals)

            global_thr, _, _ = self.get_global_threshold(all_q_vals, looseness=4)

            logger.info(
                f"Thresholding:\tglobal_thr: {round(global_thr, 2)} \tglobal_std_THR: {round(global_std_thresh, 2)}\t{'(Looks like a Xeroxed OMR)' if (global_thr == 255) else ''}"
            )

            per_omr_threshold_avg, total_q_strip_no, total_q_box_no = 0, 0, 0
            for field_block in template.field_blocks:
                block_q_strip_no = 1
                box_w, box_h = field_block.bubble_dimensions
                shift = field_block.shift
                s, d = field_block.origin, field_block.dimensions
                key = field_block.name[:3]

                for field_block_bubbles in field_block.traverse_bubbles:

                    no_outliers = all_q_std_vals[total_q_strip_no] < global_std_thresh

                    per_q_strip_threshold = self.get_local_threshold(
                        all_q_strip_arrs[total_q_strip_no],
                        global_thr,
                        no_outliers,
                        f"Mean Intensity Histogram for {key}.{field_block_bubbles[0].field_label}.{block_q_strip_no}",
                        config.outputs.show_image_level >= 6,
                    )

                    per_omr_threshold_avg += per_q_strip_threshold

                    detected_bubbles = []
                    for bubble in field_block_bubbles:
                        bubble_is_marked = (
                            per_q_strip_threshold > all_q_vals[total_q_box_no]
                        )
                        total_q_box_no += 1
                        if bubble_is_marked:
                            detected_bubbles.append(bubble)
                            x, y, field_value = (
                                bubble.x + field_block.shift,
                                bubble.y,
                                bubble.field_value,
                            )
                            cv2.rectangle(
                                final_marked,
                                (int(x + box_w / 12), int(y + box_h / 12)),
                                (
                                    int(x + box_w - box_w / 12),
                                    int(y + box_h - box_h / 12),
                                ),
                                constants.CLR_DARK_GRAY,
                                3,
                            )

                            cv2.putText(
                                final_marked,
                                str(field_value),
                                (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                constants.TEXT_SIZE,
                                (20, 20, 10),
                                int(1 + 3.5 * constants.TEXT_SIZE),
                            )
                        else:
                            cv2.rectangle(
                                final_marked,
                                (int(x + box_w / 10), int(y + box_h / 10)),
                                (
                                    int(x + box_w - box_w / 10),
                                    int(y + box_h - box_h / 10),
                                ),
                                constants.CLR_GRAY,
                                -1,
                            )

                    for bubble in detected_bubbles:
                        field_label, field_value = (
                            bubble.field_label,
                            bubble.field_value,
                        )

                        multi_marked_local = field_label in omr_response
                        omr_response[field_label] = (
                            (omr_response[field_label] + field_value)
                            if multi_marked_local
                            else field_value
                        )

                        multi_marked = multi_marked or multi_marked_local

                    if len(detected_bubbles) == 0:
                        field_label = field_block_bubbles[0].field_label
                        omr_response[field_label] = field_block.empty_val

                    block_q_strip_no += 1
                    total_q_strip_no += 1

            per_omr_threshold_avg /= total_q_strip_no
            per_omr_threshold_avg = round(per_omr_threshold_avg, 2)

            cv2.addWeighted(
                final_marked, alpha, transp_layer, 1 - alpha, 0, final_marked
            )

            if config.outputs.save_detections and save_dir is not None:
                if multi_roll:
                    save_dir = save_dir.joinpath("_MULTI_")
                image_path = str(save_dir.joinpath(name))
                ImageUtils.save_img(image_path, final_marked)

            self.append_save_img(2, final_marked)

            if save_dir is not None:
                for i in range(config.outputs.save_image_level):
                    self.save_image_stacks(i + 1, name, save_dir)

            return omr_response, final_marked, multi_marked, multi_roll

        except Exception as e:
            raise e

    @staticmethod
    def draw_template_layout(img, template, shifted=True, draw_qvals=False, border=-1):
        img = ImageUtils.resize_util(
            img, template.page_dimensions[0], template.page_dimensions[1]
        )
        final_align = img.copy()
        img_width = template.page_dimensions[0]

        for field_block in template.field_blocks:
            s, d = field_block.origin, field_block.dimensions
            box_w, box_h = field_block.bubble_dimensions
            shift = field_block.shift

            s = (img_width - s[0] - d[0], s[1])

            if shifted:
                cv2.rectangle(
                    final_align,
                    (s[0] - shift, s[1]),
                    (s[0] - shift + d[0], s[1] + d[1]),
                    constants.CLR_BLACK,
                    2,
                )
            else:
                cv2.rectangle(
                    final_align,
                    (s[0], s[1]),
                    (s[0] + d[0], s[1] + d[1]),
                    constants.CLR_BLACK,
                    2,
                )
            for field_block_bubbles in field_block.traverse_bubbles:
                for pt in field_block_bubbles:

                    x, y = (
                        (img_width - pt.x - box_w + field_block.shift, pt.y)
                        if shifted
                        else (img_width - pt.x - box_w, pt.y)
                    )
                    cv2.rectangle(
                        final_align,
                        (int(x + box_w / 10), int(y + box_h / 10)),
                        (int(x + box_w - box_w / 10), int(y + box_h - box_h / 10)),
                        constants.CLR_GRAY,
                        1,
                    )
                    if draw_qvals:
                        rect = [y, y + box_h, x, x + box_w]
                        cv2.putText(
                            final_align,
                            f"{int(cv2.mean(img[rect[0] : rect[1], rect[2] : rect[3]])[0])}",
                            (rect[2] + 2, rect[0] + (box_h * 2) // 3),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            constants.CLR_BLACK,
                            2,
                        )

            if shifted:
                text_in_px = cv2.getTextSize(
                    field_block.name, cv2.FONT_HERSHEY_SIMPLEX, constants.TEXT_SIZE, 4
                )

                cv2.putText(
                    final_align,
                    field_block.name,
                    (int(s[0] - text_in_px[0][0]), int(s[1] - text_in_px[0][1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    constants.TEXT_SIZE,
                    constants.CLR_BLACK,
                    4,
                )

        return final_align

    def get_global_threshold(
        self,
        q_vals_orig,
        plot_title=None,
        plot_show=True,
        sort_in_plot=True,
        looseness=1,
    ):
        """
        Note: Cannot assume qStrip has only-gray or only-white bg
            (in which case there is only one jump).
        So there will be either 1 or 2 jumps.
        1 Jump :
                ......
                ||||||
                ||||||  <-- risky THR
                ||||||  <-- safe THR
            ....||||||
            ||||||||||

        2 Jumps :
                ......
                |||||| <-- wrong THR
            ....||||||
            |||||||||| <-- safe THR
            ..||||||||||
            ||||||||||||

        The abstract "First LARGE GAP" is perfect for this.
        Current code is considering ONLY TOP 2 jumps(>= MIN_GAP) to be big,
            gives the smaller one

        """
        config = self.tuning_config
        PAGE_TYPE_FOR_THRESHOLD, MIN_JUMP, JUMP_DELTA = map(
            config.threshold_params.get,
            [
                "PAGE_TYPE_FOR_THRESHOLD",
                "MIN_JUMP",
                "JUMP_DELTA",
            ],
        )

        global_default_threshold = (
            constants.GLOBAL_PAGE_THRESHOLD_WHITE
            if PAGE_TYPE_FOR_THRESHOLD == "white"
            else constants.GLOBAL_PAGE_THRESHOLD_BLACK
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

        global_thr, j_low, j_high = thr1, thr1 - max1 // 2, thr1 + max1 // 2

        if plot_title:
            _, ax = plt.subplots()
            ax.bar(range(len(q_vals_orig)), q_vals if sort_in_plot else q_vals_orig)
            ax.set_title(plot_title)
            thrline = ax.axhline(global_thr, color="green", ls="--", linewidth=5)
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
        TODO: Update this documentation too-
        //No more - Assumption : Colwise background color is uniformly gray or white,
                but not alternating. In this case there is atmost one jump.

        0 Jump :
                        <-- safe THR?
            .......
            ...|||||||
            ||||||||||  <-- safe THR?
        // How to decide given range is above or below gray?
            -> global q_vals shall absolutely help here. Just run same function
                on total q_vals instead of colwise _//
        How to decide it is this case of 0 jumps

        1 Jump :
                ......
                ||||||
                ||||||  <-- risky THR
                ||||||  <-- safe THR
            ....||||||
            ||||||||||

        """
        config = self.tuning_config

        q_vals = sorted(q_vals)

        if len(q_vals) < 3:
            thr1 = (
                global_thr
                if np.max(q_vals) - np.min(q_vals) < config.threshold_params.MIN_GAP
                else np.mean(q_vals)
            )
        else:

            l = len(q_vals) - 1
            max1, thr1 = config.threshold_params.MIN_JUMP, 255
            for i in range(1, l):
                jump = q_vals[i + 1] - q_vals[i - 1]
                if jump > max1:
                    max1 = jump
                    thr1 = q_vals[i - 1] + jump / 2

            confident_jump = (
                config.threshold_params.MIN_JUMP
                + config.threshold_params.CONFIDENT_SURPLUS
            )

            if max1 < confident_jump:
                if no_outliers:

                    thr1 = global_thr
                else:

                    pass

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

    def append_save_img(self, key, img):
        if self.save_image_level >= int(key):
            self.save_img_list[key].append(img.copy())

    def save_image_stacks(self, key, filename, save_dir):
        config = self.tuning_config
        if self.save_image_level >= int(key) and self.save_img_list[key] != []:
            name = os.path.splitext(filename)[0]
            result = np.hstack(
                tuple(
                    [
                        ImageUtils.resize_util_h(img, config.dimensions.display_height)
                        for img in self.save_img_list[key]
                    ]
                )
            )
            result = ImageUtils.resize_util(
                result,
                min(
                    len(self.save_img_list[key]) * config.dimensions.display_width // 3,
                    int(config.dimensions.display_width * 2.5),
                ),
            )
            ImageUtils.save_img(f"{save_dir}stack/{name}_{str(key)}_stack.jpg", result)

    def reset_all_save_img(self):
        for i in range(self.save_image_level):
            self.save_img_list[i + 1] = []
