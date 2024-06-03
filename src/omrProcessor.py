from pathlib import Path
import cv2
from matplotlib import pyplot as plt
import numpy as np
import logging

from src import constants
from src.defaults.config import CONFIG_DEFAULTS

logger = logging.getLogger(__name__)


class ImageUtils:
    @staticmethod
    def resize_util(image, width, height):
        return cv2.resize(image, (width, height))

    @staticmethod
    def normalize_util(image):
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    @staticmethod
    def adjust_gamma(image, gamma):
        invGamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        return cv2.LUT(image, table)

    @staticmethod
    def save_img(path, image):
        cv2.imwrite(path, image)


class CLAHE_HELPER:
    @staticmethod
    def apply(image):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)


class OMRProcessor:

    def __init__(self):
        super().__init__()
        self.tuning_config = CONFIG_DEFAULTS

    def read_omr_response(self, template, image, name, save_dir=None):
        auto_align = self.tuning_config.alignment_params.auto_align
        try:
            img = self.preprocess_image(image, template)

            transp_layer = img.copy()
            final_marked = img.copy()

            morph = img.copy()

            if auto_align:
                morph = self.align_image(morph, template)

            omr_response, multi_marked = self.detect_bubbles(
                template, morph, img, final_marked, transp_layer
            )

            self.save_final_image(final_marked, save_dir, name, multi_marked)

            return (
                omr_response,
                final_marked,
                multi_marked,
                False,  # multi_roll is not modified in your code
            )

        except Exception as e:
            logger.exception("An error occurred while reading OMR response.")
            raise e

    def preprocess_image(self, image, template):
        img = image.copy()
        img = cv2.flip(img, 1)
        img = ImageUtils.resize_util(
            img, template.page_dimensions[0], template.page_dimensions[1]
        )
        if img.max() > img.min():
            img = ImageUtils.normalize_util(img)
        return img

    def align_image(self, img, template):
        # Ensure auto-align is checked before proceeding with specific steps
        if self.tuning_config.alignment_params.auto_align:
            morph = CLAHE_HELPER.apply(img)
            morph = ImageUtils.adjust_gamma(
                morph, self.tuning_config.threshold_params.GAMMA_LOW
            )
            _, morph = cv2.threshold(morph, 220, 220, cv2.THRESH_TRUNC)
            morph = ImageUtils.normalize_util(morph)

        config = self.tuning_config
        morph = CLAHE_HELPER.apply(img)
        self.append_save_img(3, morph)
        morph = ImageUtils.adjust_gamma(morph, config.threshold_params.GAMMA_LOW)
        _, morph = cv2.threshold(morph, 220, 220, cv2.THRESH_TRUNC)
        morph = ImageUtils.normalize_util(morph)
        self.append_save_img(3, morph)

        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
        morph_v = cv2.morphologyEx(morph, cv2.MORPH_OPEN, v_kernel, iterations=3)
        _, morph_v = cv2.threshold(morph_v, 200, 200, cv2.THRESH_TRUNC)
        morph_v = 255 - ImageUtils.normalize_util(morph_v)
        self.append_save_img(3, morph_v)

        morph_thr = 60
        _, morph_v = cv2.threshold(morph_v, morph_thr, 255, cv2.THRESH_BINARY)
        morph_v = cv2.erode(morph_v, np.ones((5, 5), np.uint8), iterations=2)
        self.append_save_img(3, morph_v)
        self.append_save_img(6, morph_v)

        self.align_field_blocks(template, morph_v)
        return morph_v

    def align_field_blocks(self, template, morph_v):
        config = self.tuning_config
        for field_block in template.field_blocks:
            s, d = field_block.origin, field_block.dimensions
            match_col, max_steps, align_stride, thk = map(
                config.alignment_params.get,
                ["match_col", "max_steps", "stride", "thickness"],
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

    def detect_bubbles(self, template, morph, img, final_marked, transp_layer):
        config = self.tuning_config
        alpha = 0.65
        omr_response = {}
        multi_marked = False
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
            f"Thresholding:\tglobal_thr: {round(global_thr, 2)} \tglobal_std_THR: {round(global_std_thresh, 2)}\t"
            f"{'(Looks like a Xeroxed OMR)' if (global_thr == 255) else ''}"
        )

        per_omr_threshold_avg, total_q_strip_no, total_q_box_no = 0, 0, 0
        for field_block in template.field_blocks:
            block_q_strip_no = 1
            box_w, box_h = field_block.bubble_dimensions
            shift = field_block.shift
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
                        self.mark_bubble(final_marked, x, y, box_w, box_h, field_value)
                    else:
                        self.mark_empty_bubble(final_marked, x, y, box_w, box_h)

                for bubble in detected_bubbles:
                    field_label, field_value = (bubble.field_label, bubble.field_value)
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

        cv2.addWeighted(final_marked, alpha, transp_layer, 1 - alpha, 0, final_marked)

        return omr_response, multi_marked

    def mark_bubble(self, final_marked, x, y, box_w, box_h, field_value):
        cv2.rectangle(
            final_marked,
            (int(x + box_w / 12), int(y + box_h / 12)),
            (int(x + box_w - box_w / 12), int(y + box_h - box_h / 12)),
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

    def mark_empty_bubble(self, final_marked, x, y, box_w, box_h):
        cv2.rectangle(
            final_marked,
            (int(x + box_w / 10), int(y + box_h / 10)),
            (int(x + box_w - box_w / 10), int(y + box_h - box_h / 10)),
            constants.CLR_GRAY,
            -1,
        )

    def save_final_image(self, final_marked, save_dir, name, multi_marked):
        if multi_marked:
            save_dir = save_dir.joinpath("_MULTI_")
        image_path = str(save_dir.joinpath(name))
        ImageUtils.save_img(image_path, final_marked)
        for i in range(self.tuning_config.outputs.save_image_level):
            self.save_image_stacks(i + 1, name, save_dir)

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
