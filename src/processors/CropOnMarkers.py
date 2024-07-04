import os
import cv2
import numpy as np
from logger import logger
from defaults.config import CONFIG_DEFAULTS
from utils.image import ImageUtils


class CropOnMarkers:
    def __init__(self, marker_ops=None):
        if marker_ops is None:
            marker_ops = {
                "relativePath": "../constants/marker_image.jpg",
                "sheetToMarkerWidthRatio": 21,
                "min_matching_threshold": 0.3,
                "max_matching_variation": 0.41,
                "marker_rescale_range": (35, 100),
                "marker_rescale_steps": 10,
                "apply_erode_subtract": True,
            }

        self.tuning_config = CONFIG_DEFAULTS
        self.marker_path = os.path.join(
            os.path.dirname(__file__), marker_ops["relativePath"]
        )
        self.min_matching_threshold = marker_ops["min_matching_threshold"]
        self.max_matching_variation = marker_ops["max_matching_variation"]
        self.marker_rescale_range = tuple(marker_ops["marker_rescale_range"])
        self.marker_rescale_steps = marker_ops["marker_rescale_steps"]
        self.apply_erode_subtract = marker_ops["apply_erode_subtract"]
        self.threshold_circles = []
        self.marker = self.load_marker(marker_ops, self.tuning_config)

    def __str__(self):
        return self.marker_path

    def exclude_files(self):
        return [self.marker_path]

    def apply_filter(self, image, file_path):
        eroded_image = self._apply_erode_subtract(image)
        quads, origins = self._divide_image_into_quadrants(eroded_image)
        self._draw_dividers(eroded_image)

        best_scale, all_max_t = self._get_best_match(eroded_image)
        if best_scale is None:
            return None

        optimal_marker = ImageUtils.resize_util_h(
            self.marker, u_height=int(self.marker.shape[0] * best_scale)
        )
        centres, success = self._find_marker_centres(
            quads, optimal_marker, origins, file_path, all_max_t, eroded_image
        )
        if not success:
            return None

        logger.info(f"Optimal Scale: {best_scale}")
        self.threshold_circles.append(sum(max_t for _, max_t in centres) / 4)

        transformed_image = ImageUtils.four_point_transform(
            image, np.array([pt for pt, _ in centres])
        )
        return transformed_image

    def load_marker(self, marker_ops, config):
        if not os.path.exists(self.marker_path):
            logger.error(
                "Marker not found at path provided in template:", self.marker_path
            )
            exit(31)

        marker = cv2.imread(self.marker_path, cv2.IMREAD_GRAYSCALE)
        sheet_to_marker_ratio = (
            config.dimensions.processing_width / marker_ops["sheetToMarkerWidthRatio"]
        )
        marker = ImageUtils.resize_util(marker, sheet_to_marker_ratio)
        marker = cv2.GaussianBlur(marker, (5, 5), 0)
        marker = cv2.normalize(
            marker, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        )

        if self.apply_erode_subtract:
            marker -= cv2.erode(marker, kernel=np.ones((5, 5)), iterations=5)

        return marker

    def _apply_erode_subtract(self, image):
        if self.apply_erode_subtract:
            return ImageUtils.normalize_util(image)
        else:
            return ImageUtils.normalize_util(
                image - cv2.erode(image, kernel=np.ones((5, 5)), iterations=5)
            )

    def _divide_image_into_quadrants(self, image):
        h, w = image.shape[:2]
        midh, midw = h // 3, w // 2
        quads = {
            0: image[0:midh, 0:midw],
            1: image[0:midh, midw:w],
            2: image[midh:h, 0:midw],
            3: image[midh:h, midw:w],
        }
        origins = [[0, 0], [midw, 0], [0, midh], [midw, midh]]
        return quads, origins

    def _draw_dividers(self, image):
        h, w = image.shape[:2]
        midh, midw = h // 3, w // 2
        image[:, midw : midw + 2] = 255
        image[midh : midh + 2, :] = 255

    def _get_best_match(self, image):
        descent_per_step = (
            self.marker_rescale_range[1] - self.marker_rescale_range[0]
        ) // self.marker_rescale_steps
        best_scale, all_max_t = None, 0

        for scale in np.arange(
            self.marker_rescale_range[1],
            self.marker_rescale_range[0],
            -descent_per_step,
        ):
            s = scale / 100.0
            if s == 0.0:
                continue
            rescaled_marker = ImageUtils.resize_util_h(
                self.marker, u_height=int(self.marker.shape[0] * s)
            )
            res = cv2.matchTemplate(image, rescaled_marker, cv2.TM_CCOEFF_NORMED)
            max_t = res.max()

            if max_t > all_max_t:
                best_scale, all_max_t = s, max_t

        if all_max_t < self.min_matching_threshold:
            logger.warning(
                "Template matching too low! Consider rechecking preProcessors applied before this."
            )

        if best_scale is None:
            logger.warning(
                "No matchings for given scaleRange: %s", self.marker_rescale_range
            )
        return best_scale, all_max_t

    def _find_marker_centres(
        self, quads, marker, origins, file_path, all_max_t, image_eroded_sub
    ):
        centres = []
        success = True
        quarter_match_log = "Matching Marker:  "

        for k in range(4):
            res = cv2.matchTemplate(quads[k], marker, cv2.TM_CCOEFF_NORMED)
            max_t = res.max()
            quarter_match_log += f"Quarter{str(k + 1)}: {str(round(max_t, 3))}\t"

            if (
                max_t < self.min_matching_threshold
                or abs(all_max_t - max_t) >= self.max_matching_variation
            ):
                logger.error(
                    "%s\nError: No circle found in Quad %d\n\t min_matching_threshold %f\t max_matching_variation %f\t max_t %f\t all_max_t %f",
                    file_path,
                    k + 1,
                    self.min_matching_threshold,
                    self.max_matching_variation,
                    max_t,
                    all_max_t,
                )
                success = False
                break

            pt = np.argwhere(res == max_t)[0]
            pt = [pt[1], pt[0]]
            pt[0] += origins[k][0]
            pt[1] += origins[k][1]
            centres.append(
                ([pt[0] + marker.shape[1] // 2, pt[1] + marker.shape[0] // 2], max_t)
            )
            self._draw_marker_rectangle(image_eroded_sub, pt, marker.shape)

        logger.info(quarter_match_log)
        return centres, success

    def _draw_marker_rectangle(self, image, pt, marker_shape):
        _h, w = marker_shape[:2]
        image = cv2.rectangle(
            image,
            tuple(pt),
            (pt[0] + w, pt[1] + _h),
            (50, 50, 50) if self.apply_erode_subtract else (155, 155, 155),
            4,
        )
