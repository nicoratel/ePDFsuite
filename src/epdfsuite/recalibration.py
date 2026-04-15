import numpy as np
from skimage import measure
from pyFAI.utils.ellipse import fit_ellipse as pyfai_fit_ellipse
import matplotlib.pyplot as plt

def _fit_circle_algebraic(y_pts, x_pts):
    """Algebraic (Kåsa) circle fit on a set of 2-D points.

    Solves the linearised problem ``Dx + Ey + F = -(x²+y²)`` in the
    least-squares sense.  More constrained and numerically stabler than
    a general ellipse fit, especially on partial arcs.

    Parameters
    ----------
    y_pts, x_pts : array_like
        Coordinates of the edge pixels (slow / fast axes).

    Returns
    -------
    x_c, y_c : float
        Centre coordinates.
    radius : float
        Fitted circle radius in pixels.
    """
    x = np.asarray(x_pts, dtype=float)
    y = np.asarray(y_pts, dtype=float)
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x ** 2 + y ** 2)
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    D, E, F = result
    x_c = -D / 2.0
    y_c = -E / 2.0
    r2 = x_c ** 2 + y_c ** 2 - F
    if r2 <= 0:
        raise ValueError("Circle fit produced a negative radius squared. Check input points.")
    return x_c, y_c, float(np.sqrt(r2))





# ============================================================
# Iso-intensity contour method (amorphous halo + large beamstop)
# ============================================================

def recalibrate_from_isocurve(image, n_levels=7, level_range=(0.15, 0.75),
                               min_points=40, fit_circle=True,
                               rms_rel_max=0.15, min_arc_deg=60.0,
                               cluster_window=15.0,
                               initial_center=None, max_center_offset=None,
                               mask=None, plot=False):
    """Find the beam centre from iso-intensity contours of an amorphous halo.

    For amorphous specimens the scattered intensity decreases monotonically
    from the direct beam, so iso-intensity lines are approximately circular
    arcs centred on the beam.  This method:

    1. Computes intensity percentiles on *valid* (unmasked) pixels only,
       then sets masked pixels to 0 so they do not generate contours.
    2. Samples *n_levels* intensity levels across *level_range*.
    3. Extracts iso-intensity contours with ``skimage.measure.find_contours``.
    4. Discards contour points that fall inside the masked region.
    5. Fits a circle (Kasa algebraic method) to each surviving contour.
    6. Filters by circularity: keeps only fits where ``rms / R < rms_rel_max``
       (rejects non-circular contours such as mask edges or noise).
    7. Filters by arc span: keeps only fits whose angular coverage around
       the fitted centre exceeds *min_arc_deg* (rejects very short arcs that
       produce biased Kasa estimates).
    8. If *initial_center* is provided, additionally discards fits whose
       centre is farther than *max_center_offset* pixels from that estimate.
    9. **Clusters** the surviving centre estimates: computes a robust median,
       then discards outliers whose centre deviates from it by more than
       *cluster_window* pixels.
    10. Returns the **median** of the final cluster.

    Parameters
    ----------
    image : ndarray
        2-D image as a numpy array.
    n_levels : int, optional
        Number of iso-intensity levels to sample.  Default is 7.
    level_range : tuple of float, optional
        ``(low, high)`` as fractions of the visible (unmasked) intensity
        range after hot-pixel clipping.  Default is ``(0.15, 0.75)``.
    min_points : int, optional
        Minimum number of unmasked contour points required to attempt a
        fit.  Default is 40.
    fit_circle : bool, optional
        If ``True`` (default), fit a circle (3 parameters, more stable
        on partial arcs).  If ``False``, fit a general ellipse.
    rms_rel_max : float, optional
        Maximum allowed normalised RMS residual ``rms / R`` to accept a
        contour as circular.  Default is ``0.15`` (15 % of radius).
        Increase if the halo is slightly elliptical; decrease to be stricter.
    min_arc_deg : float, optional
        Minimum angular span (degrees) of the arc around the fitted centre
        required to accept the contour.  Default is ``60.0``.  Short arcs
        (beamstop edge, noise patch) are rejected.
    initial_center : tuple of float or None, optional
        Rough centre ``(x, y)`` in pixels used to filter out geometrically
        implausible circle fits.  When ``None`` no distance filtering is
        applied.
    cluster_window : float, optional
        Radius (pixels) of the final clustering window.  After all per-contour
        fits have passed the circularity and arc-span filters, a robust median
        is computed and any estimate farther than *cluster_window* pixels from
        it is discarded as an outlier before computing the final centre.
        Default is ``15.0``.
    max_center_offset : float or None, optional
        Maximum allowed distance (pixels) between a per-contour fitted
        centre and *initial_center*.  Defaults to 25 % of the smallest
        image dimension when *initial_center* is provided.
    mask : ndarray of bool or None, optional
        Boolean mask where ``True`` marks *invalid* pixels (beamstop,
        detector edge, etc.).
    plot : bool, optional
        Show a diagnostic figure.  Default is ``False``.

    Returns
    -------
    x_c : float
        X coordinate of the beam centre (pixels).
    y_c : float
        Y coordinate of the beam centre (pixels).
    """
    img = image.astype(float)
    h, w = img.shape

    # Boolean mask of invalid pixels
    invalid = np.asarray(mask, dtype=bool) if mask is not None \
        else np.zeros(img.shape, dtype=bool)

    # --- Normalise using only VALID pixels ---
    valid_vals = img[~invalid]
    if valid_vals.size == 0:
        raise ValueError("recalibrate_from_isocurve: mask covers entire image.")
    vmin_clip = np.percentile(valid_vals, 1)
    vmax_clip = np.percentile(valid_vals, 99)
    img_norm = np.clip((img - vmin_clip) / (vmax_clip - vmin_clip + 1e-10),
                       0.0, 1.0)
    # Masked pixels → 0 so find_contours does not generate artefact contours
    # along the mask boundary
    img_norm[invalid] = 0.0

    levels = np.linspace(level_range[0], level_range[1], n_levels)

    # Distance filter based on initial centre
    if initial_center is not None:
        x_rough, y_rough = float(initial_center[0]), float(initial_center[1])
        _offset_limit = max_center_offset if max_center_offset is not None \
            else min(h, w) * 0.25
    else:
        x_rough, y_rough = None, None
        _offset_limit = None

    # Accepted contours (used for median)
    x_centers, y_centers, rms_list = [], [], []
    # Rejected contours stored for diagnostic plot only
    # Each entry: (xs_plot, ys_plot, reason_string)
    _rejected = []

    for level in levels:
        for cnt in measure.find_contours(img_norm, level):
            # find_contours returns (row, col) == (y, x)
            ys_all = cnt[:, 0]
            xs_all = cnt[:, 1]

            # Remove points that fall inside the mask
            yi = np.clip(np.round(ys_all).astype(int), 0, h - 1)
            xi = np.clip(np.round(xs_all).astype(int), 0, w - 1)
            valid_pts = ~invalid[yi, xi]
            ys, xs = ys_all[valid_pts], xs_all[valid_pts]

            if len(xs) < min_points:
                continue

            try:
                if fit_circle:
                    x_c_i, y_c_i, r_i = _fit_circle_algebraic(ys, xs)
                    r_pred = np.sqrt((xs - x_c_i) ** 2 + (ys - y_c_i) ** 2)
                    rms = float(np.std(r_pred - r_i))
                else:
                    params = pyfai_fit_ellipse(np.column_stack([ys, xs]))
                    y_c_i, x_c_i = params[0], params[1]
                    r_i = np.sqrt((xs - x_c_i) ** 2 + (ys - y_c_i) ** 2).mean()
                    rms = 0.0

                # Reject centres outside image bounds
                if not (0 <= x_c_i < w and 0 <= y_c_i < h):
                    _rejected.append((xs, ys, 'out of bounds'))
                    continue

                # --- Circularity filter: rms / R ---
                rms_rel = rms / r_i if r_i > 0 else np.inf
                if rms_rel > rms_rel_max:
                    _rejected.append((xs, ys, f'rms/R={rms_rel:.2f}'))
                    continue

                # --- Arc span filter ---
                angles = np.arctan2(ys - y_c_i, xs - x_c_i)
                angles_sorted = np.sort(angles)
                gaps = np.diff(angles_sorted)
                wrap_gap = 2.0 * np.pi - (angles_sorted[-1] - angles_sorted[0])
                largest_gap = max(float(gaps.max()), wrap_gap)
                arc_span_deg = float(np.degrees(2.0 * np.pi - largest_gap))
                if arc_span_deg < min_arc_deg:
                    _rejected.append((xs, ys, f'arc={arc_span_deg:.0f}°'))
                    continue

                # --- Distance from initial centre ---
                if x_rough is not None:
                    dist = np.sqrt((x_c_i - x_rough) ** 2 + (y_c_i - y_rough) ** 2)
                    if dist > _offset_limit:
                        _rejected.append((xs, ys, 'too far'))
                        continue

                x_centers.append(x_c_i)
                y_centers.append(y_c_i)
                rms_list.append(rms)
            except Exception:
                continue

    if len(x_centers) == 0:
        raise ValueError(
            "recalibrate_from_isocurve: no valid contour fit found. "
            "Try adjusting level_range, n_levels, min_points, rms_rel_max, "
            "min_arc_deg, or increase max_center_offset."
        )

    x_centers = np.array(x_centers)
    y_centers = np.array(y_centers)
    rms_arr = np.array(rms_list)

    # --- Cluster-based outlier rejection ---
    # 1. Robust initial estimate via median
    x_med0 = float(np.median(x_centers))
    y_med0 = float(np.median(y_centers))
    # 2. Keep only centres within cluster_window of that estimate
    dist_to_med = np.sqrt((x_centers - x_med0) ** 2 + (y_centers - y_med0) ** 2)
    in_cluster = dist_to_med <= cluster_window
    if in_cluster.sum() == 0:          # safety fallback
        in_cluster = np.ones(len(x_centers), dtype=bool)
    # 3. Final centre = median of the cluster
    x_c = float(np.median(x_centers[in_cluster]))
    y_c = float(np.median(y_centers[in_cluster]))

    if plot:
        vmin_d = np.percentile(valid_vals, 1)
        vmax_d = np.percentile(valid_vals, 99)
        img_display = img.copy()
        img_display[invalid] = np.nan
        cmap_w = plt.get_cmap('gray').copy()
        cmap_w.set_bad(color='white')

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Panel 1: image + contours
        axes[0].imshow(img_display, cmap=cmap_w, vmin=vmin_d, vmax=vmax_d,
                       origin='upper')

        # Draw rejected contours in light grey
        _rej_drawn = False
        for (xs_r, ys_r, _reason) in _rejected:
            axes[0].plot(xs_r, ys_r, color='#aaaaaa', lw=0.6, alpha=0.4,
                         label='_' if _rej_drawn else 'Rejected (non-circular / short arc)')
            _rej_drawn = True

        # Draw accepted contours in colour (cycle through tab10)
        cmap_tab = plt.get_cmap('tab10')
        _seen_levels = {}
        for level in levels:
            for cnt in measure.find_contours(img_norm, level):
                yi = np.clip(np.round(cnt[:, 0]).astype(int), 0, h - 1)
                xi = np.clip(np.round(cnt[:, 1]).astype(int), 0, w - 1)
                v = ~invalid[yi, xi]
                if v.sum() < min_points:
                    continue
                # Check if this contour was accepted: refit and recheck
                ys_v, xs_v = cnt[v, 0], cnt[v, 1]
                try:
                    xci, yci, ri = _fit_circle_algebraic(ys_v, xs_v)
                    r_pred = np.sqrt((xs_v - xci) ** 2 + (ys_v - yci) ** 2)
                    rms_i = float(np.std(r_pred - ri))
                    rr = rms_i / ri if ri > 0 else np.inf
                    ang = np.arctan2(ys_v - yci, xs_v - xci)
                    ang_s = np.sort(ang)
                    lg = max(float(np.diff(ang_s).max()),
                             2 * np.pi - (ang_s[-1] - ang_s[0]))
                    span = float(np.degrees(2 * np.pi - lg))
                    in_bounds = (0 <= xci < w and 0 <= yci < h)
                    near = True
                    if x_rough is not None:
                        near = np.sqrt((xci - x_rough) ** 2 + (yci - y_rough) ** 2) <= _offset_limit
                    accepted = in_bounds and rr <= rms_rel_max and span >= min_arc_deg and near
                except Exception:
                    accepted = False

                if accepted:
                    idx = list(levels).index(level) if level in list(levels) else 0
                    color = cmap_tab(idx % 10)
                    lbl = f'level={level:.2f}' if level not in _seen_levels else '_'
                    _seen_levels[level] = True
                    axes[0].plot(xs_v, ys_v, color=color, lw=1.2, alpha=0.85,
                                 label=lbl)

        if x_rough is not None:
            axes[0].plot(x_rough, y_rough, 'b+', markersize=14,
                         markeredgewidth=2,
                         label=f'Initial ({x_rough:.0f}, {y_rough:.0f})')
        axes[0].plot(x_c, y_c, 'r+', markersize=16, markeredgewidth=2.5,
                     label=f'Centre ({x_c:.1f}, {y_c:.1f})')
        axes[0].set_title('Iso-intensity contours', fontsize=13)
        axes[0].legend(fontsize=8, loc='lower right')
        axes[0].grid(True, alpha=0.3)

        # Panel 2: cluster members coloured by RMS, outliers as grey ×
        if (~in_cluster).any():
            axes[1].scatter(x_centers[~in_cluster], y_centers[~in_cluster],
                            color='#aaaaaa', s=40, marker='x', zorder=2, linewidths=1.5,
                            label=f'Outliers (>{cluster_window:.0f} px) '
                                  f'[{int((~in_cluster).sum())}]')
        sc = axes[1].scatter(x_centers[in_cluster], y_centers[in_cluster],
                             c=rms_arr[in_cluster], cmap='RdYlGn_r', s=40, zorder=3,
                             label=f'Cluster [{int(in_cluster.sum())}]')
        plt.colorbar(sc, ax=axes[1], label='RMS residual (px)')
        if x_rough is not None:
            axes[1].plot(x_rough, y_rough, 'b+', markersize=14,
                         markeredgewidth=2,
                         label=f'Initial ({x_rough:.0f}, {y_rough:.0f})')
        axes[1].plot(x_c, y_c, 'r*', markersize=18, zorder=4,
                     label=f'Median ({x_c:.1f}, {y_c:.1f})')
        axes[1].set_title('Centre estimates – cluster analysis', fontsize=13)
        axes[1].legend(fontsize=9)
        axes[1].set_aspect('equal', 'box')
        axes[1].invert_yaxis()
        axes[1].grid(True, alpha=0.3)

        for ax in axes:
            ax.set_xlabel('X (pixels)', fontsize=11)
            ax.set_ylabel('Y (pixels)', fontsize=11)
        n_rej = len(_rejected)
        n_out = int((~in_cluster).sum())
        fig.suptitle(
            f'recalibrate_from_isocurve  |  cluster: {int(in_cluster.sum())} pts'
            f'  |  outliers: {n_out}  |  shape-rejected: {n_rej}'
            f'  |  centre = ({x_c:.2f}, {y_c:.2f}) px',
            fontsize=11
        )
        fig.tight_layout()
        plt.show()

    return x_c, y_c
