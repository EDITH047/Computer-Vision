"""
cloud_tracker.py
----------------
Python port of the C++ CloudTracker class (CloudTracker.h + CloudTracker.cpp).

Original C++ project: Satellite Image Cloud Detection and Wind Speed Estimation
Python translation: full 1-to-1 functional equivalence using opencv-python + numpy.
"""

import cv2
import numpy as np


class CloudTracker:
    """
    Satellite Image Cloud Detection and Wind Speed Estimator.

    Workflow
    --------
    1. load_images(path1, path2)  — load two satellite frames
    2. process(time_difference)   — run the full pipeline
    3. show_results()             — display OpenCV windows + print statistics

    All internal methods mirror the private helpers of the C++ class.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────────────────

    def __init__(self) -> None:
        # Source images (colour + greyscale)
        self._img1_color: np.ndarray | None = None
        self._img2_color: np.ndarray | None = None
        self._img1_gray:  np.ndarray | None = None
        self._img2_gray:  np.ndarray | None = None

        # Binary cloud masks (uint8, 0 or 255)
        self._cloud_mask1: np.ndarray | None = None
        self._cloud_mask2: np.ndarray | None = None

        # Output visualisations
        self._highlighted_clouds:  np.ndarray | None = None
        self._map_overlay:         np.ndarray | None = None   # static text/borders
        self._flow:                np.ndarray | None = None   # 2-channel float32
        self._wind_speed_map:      np.ndarray | None = None   # float32
        self._flow_visualization:  np.ndarray | None = None

        # Statistics
        self._cloud_coverage_pct1: float = 0.0
        self._cloud_coverage_pct2: float = 0.0
        self._avg_wind_speed:      float = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def load_images(self, path1: str, path2: str) -> bool:
        """
        Load two satellite images from disk.

        Parameters
        ----------
        path1 : str  Path to the first image (time T1).
        path2 : str  Path to the second image (time T2).

        Returns
        -------
        bool  True on success, False if either image could not be loaded.
        """
        self._img1_color = cv2.imread(path1)
        self._img2_color = cv2.imread(path2)

        if self._img1_color is None or self._img2_color is None:
            print("Error: Could not load one or both images.")
            return False

        # Ensure both frames are the same resolution (resize T2 to match T1)
        if self._img1_color.shape != self._img2_color.shape:
            h, w = self._img1_color.shape[:2]
            self._img2_color = cv2.resize(self._img2_color, (w, h))

        # Convert to greyscale for processing
        self._img1_gray = cv2.cvtColor(self._img1_color, cv2.COLOR_BGR2GRAY)
        self._img2_gray = cv2.cvtColor(self._img2_color, cv2.COLOR_BGR2GRAY)

        return True

    def process(self, time_difference: float) -> None:
        """
        Run the complete cloud-tracking pipeline.

        Parameters
        ----------
        time_difference : float
            Time elapsed between the two satellite passes (hours, or any
            consistent unit).  Used to convert pixel displacement → wind speed.
        """
        if self._img1_gray is None or self._img2_gray is None:
            return

        # Steps 3-5: Gaussian blur + binary threshold → cloud masks
        self._preprocess_and_mask(self._img1_gray, mask_id=1)
        self._preprocess_and_mask(self._img2_gray, mask_id=2)

        # Step 6: Detect and remove static map overlays (borders, text labels)
        self._extract_overlays()

        # Step 7: Cloud coverage percentages
        total = self._cloud_mask1.size
        self._cloud_coverage_pct1 = (np.count_nonzero(self._cloud_mask1) / total) * 100.0
        self._cloud_coverage_pct2 = (np.count_nonzero(self._cloud_mask2) / total) * 100.0

        # Prepare highlighted-clouds canvas (coloured by wind speed in show_results)
        self._highlighted_clouds = self._img1_color.copy()

        # Steps 7-8: Dense Farneback optical flow
        self._compute_optical_flow()

        # Step 9: Wind speed from pixel displacement / time
        self._estimate_wind_speed(time_difference)

        # Prepare motion-vector overlay (red arrows, one per 16-px grid cell)
        self._draw_motion_vectors(step=16, color=(0, 0, 255))

    def show_results(self) -> None:
        """
        Display three OpenCV windows and print statistics to the console.

        Windows
        -------
        1. Original Image 1
        2. Highlighted Clouds   (jet colour-map keyed by wind speed + legend)
        3. Wind Direction Vectors (red arrows + yellow map overlays)
        """
        if self._img1_color is None:
            return

        # ── Wind-speed colour map ────────────────────────────────────────────
        wind_vis = cv2.normalize(
            self._wind_speed_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1
        )
        wind_vis = cv2.applyColorMap(wind_vis, cv2.COLORMAP_JET)

        # Apply cloud mask so only cloud pixels show the heat-map
        masked_speed = np.zeros_like(wind_vis)
        masked_speed[self._cloud_mask1 > 0] = wind_vis[self._cloud_mask1 > 0]

        # Colour highlighted clouds by wind speed
        self._highlighted_clouds[self._cloud_mask1 > 0] = \
            wind_vis[self._cloud_mask1 > 0]

        # ── Legend ───────────────────────────────────────────────────────────
        self._draw_legend(self._highlighted_clouds)

        # ── Restore yellow map overlays on both views ─────────────────────────
        self._flow_visualization[self._map_overlay > 0] = (0, 255, 255)
        self._highlighted_clouds[self._map_overlay > 0]  = (0, 255, 255)

        # ── Display windows ──────────────────────────────────────────────────
        cv2.imshow("1. Original Image 1",       self._img1_color)
        cv2.imshow("2. Highlighted Clouds",     self._highlighted_clouds)
        cv2.imshow("3. Wind Direction Vectors", self._flow_visualization)

        # ── Statistics ───────────────────────────────────────────────────────
        print("--- Statistics ---")
        print(f"Cloud Coverage Image 1: {self._cloud_coverage_pct1:.2f} %")
        print(f"Cloud Coverage Image 2: {self._cloud_coverage_pct2:.2f} %")
        print(f"Average Wind Speed:     {self._avg_wind_speed:.4f} km/hr")
        print("------------------")
        print("Press any key in the image windows to exit...")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ──────────────────────────────────────────────────────────────────────────
    # Private Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _preprocess_and_mask(self, src_gray: np.ndarray, mask_id: int) -> None:
        """
        Apply Gaussian blur then binary threshold to isolate bright cloud pixels.

        C++ equivalent: CloudTracker::preprocessAndMask()

        Parameters
        ----------
        src_gray : np.ndarray  Greyscale source frame.
        mask_id  : int         1 → stored in _cloud_mask1, 2 → _cloud_mask2.
        """
        blurred = cv2.GaussianBlur(src_gray, (5, 5), 0)
        # Threshold 160: clouds are bright-white in true-colour satellite imagery;
        # land and ocean are darker and fall below this value.
        _, mask = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)
        if mask_id == 1:
            self._cloud_mask1 = mask
        else:
            self._cloud_mask2 = mask

    def _extract_overlays(self) -> None:
        """
        Detect static map overlays (country borders, text labels) and remove
        them from the cloud masks so they are not mistaken for clouds.

        Algorithm (mirrors C++ CloudTracker::extractOverlays):
        1. Pixels whose intensity barely changes between frames → static.
        2. Static AND bright → candidate overlay pixel.
        3. Connected-component analysis separates small/thin components
           (text, borders) from large blobs (real clouds).
        4. Dilate the overlay mask to cover anti-aliased edges.
        5. Subtract the overlay from both cloud masks.
        """
        # Step 1: per-pixel absolute difference
        diff = cv2.absdiff(self._img1_gray, self._img2_gray)

        # Static pixels: difference < 8
        _, static_mask = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY_INV)

        # Bright pixels: intensity > 170 (text / borders are rendered white)
        _, bright_mask = cv2.threshold(self._img1_gray, 170, 255, cv2.THRESH_BINARY)

        # Candidate overlay = static AND bright
        candidate = cv2.bitwise_and(static_mask, bright_mask)

        # Step 2: Connected component analysis
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate)
        self._map_overlay = np.zeros(candidate.shape, dtype=np.uint8)

        for i in range(1, num_labels):           # skip label 0 (background)
            area   = int(stats[i, cv2.CC_STAT_AREA])
            w      = int(stats[i, cv2.CC_STAT_WIDTH])
            h      = int(stats[i, cv2.CC_STAT_HEIGHT])
            bbox   = float(w * h)
            fill   = (area / bbox) if bbox > 0.0 else 1.0

            # Classify as text/border (True) vs. real cloud blob (False)
            is_text = (
                area < 5000                                    # small char
                or ((w > 10 * h or h > 10 * w) and fill < 0.3)  # thin line
                or (area < 20000 and fill < 0.4)              # sparse group
            )
            if is_text:
                self._map_overlay[labels == i] = 255

        # Step 3: Dilate to cover anti-aliased edges around text
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(self._map_overlay, kernel)

        # Step 4: Remove overlay pixels from cloud masks
        not_overlay = cv2.bitwise_not(dilated)
        self._cloud_mask1 = cv2.bitwise_and(self._cloud_mask1, not_overlay)
        self._cloud_mask2 = cv2.bitwise_and(self._cloud_mask2, not_overlay)

    def _compute_optical_flow(self) -> None:
        """
        Calculate dense optical flow using the Farneback method.

        C++ equivalent: CloudTracker::computeOpticalFlow()
        Parameters match the C++ call exactly:
          pyr_scale=0.5, levels=3, winsize=15, iterations=3,
          poly_n=5, poly_sigma=1.2, flags=0
        """
        self._flow = cv2.calcOpticalFlowFarneback(
            self._img1_gray,
            self._img2_gray,
            None,           # flow output (None → allocate new)
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

    def _estimate_wind_speed(self, time_difference: float) -> None:
        """
        Compute per-pixel wind speed from optical-flow magnitude.

        wind_speed[y, x] = ||flow[y, x]|| / time_difference
        Only computed for pixels that are cloud in mask1.

        C++ equivalent: CloudTracker::estimateWindSpeed()
        """
        if time_difference <= 0.0:
            time_difference = 1.0

        h, w = self._flow.shape[:2]
        self._wind_speed_map = np.zeros((h, w), dtype=np.float32)

        # Vectorised computation (replaces the C++ pixel loop)
        fx = self._flow[..., 0]   # x-component
        fy = self._flow[..., 1]   # y-component
        magnitude = np.sqrt(fx ** 2 + fy ** 2)
        speed = (magnitude / time_difference).astype(np.float32)

        cloud_mask = self._cloud_mask1 > 0
        self._wind_speed_map[cloud_mask] = speed[cloud_mask]

        cloud_speeds = speed[cloud_mask]
        self._avg_wind_speed = float(cloud_speeds.mean()) if cloud_speeds.size > 0 else 0.0

    def _draw_motion_vectors(self, step: int, color: tuple) -> None:
        """
        Draw arrow vectors on a copy of image 1 to show cloud motion direction.

        Arrows are drawn only where cloud_mask1 is non-zero, sampled every
        `step` pixels in both x and y directions.

        C++ equivalent: CloudTracker::drawMotionVectors()
        """
        vis = self._img1_color.copy()
        h, w = self._flow.shape[:2]

        for y in range(0, h, step):
            for x in range(0, w, step):
                if self._cloud_mask1[y, x] > 0:
                    fx, fy = self._flow[y, x]
                    pt1 = (x, y)
                    pt2 = (int(round(x + fx)), int(round(y + fy)))
                    cv2.arrowedLine(
                        vis, pt1, pt2, color,
                        thickness=1,
                        lineType=cv2.LINE_AA,
                        shift=0,
                        tipLength=0.3,
                    )

        self._flow_visualization = vis

    def _draw_legend(self, image: np.ndarray) -> None:
        """
        Draw a horizontal Jet-colormap wind-speed legend at the bottom-centre
        of `image`, with Low / Med / High labels and a title.

        C++ equivalent: CloudTracker::drawLegend()
        """
        legend_width  = 200
        legend_height = 20
        margin        = 20

        sx = (image.shape[1] - legend_width) // 2
        sy = image.shape[0] - legend_height - margin - 30  # room for text below

        if sx < 0 or sy < 0:
            return  # image too small

        # Colour gradient bar
        gradient  = np.arange(legend_width, dtype=np.uint8).reshape(1, legend_width)
        color_bar = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
        color_bar = cv2.resize(color_bar, (legend_width, legend_height))
        image[sy : sy + legend_height, sx : sx + legend_width] = color_bar

        # White outline
        cv2.rectangle(
            image,
            (sx, sy),
            (sx + legend_width, sy + legend_height),
            (255, 255, 255),
            1,
        )

        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5

        # Labels: Low (left), Med (centre), High (right) — drawn with shadow
        labels = [
            ("Low",  sx),
            ("Med",  sx + legend_width // 2 - 15),
            ("High", sx + legend_width - 35),
        ]
        for text, ox in labels:
            # Shadow (black, +1 px offset)
            cv2.putText(image, text, (ox + 1, sy + legend_height + 21),
                        font, scale, (0, 0, 0), 1, cv2.LINE_AA)
            # Foreground (white)
            cv2.putText(image, text, (ox, sy + legend_height + 20),
                        font, scale, (255, 255, 255), 1, cv2.LINE_AA)

        title = "Wind Speed (km/hr)"
        title_x = sx + legend_width // 2 - 65
        # Shadow
        cv2.putText(image, title, (title_x + 1, sy - 9),
                    font, scale, (0, 0, 0), 1, cv2.LINE_AA)
        # Foreground
        cv2.putText(image, title, (title_x, sy - 10),
                    font, scale, (255, 255, 255), 1, cv2.LINE_AA)
