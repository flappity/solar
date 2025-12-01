#!/usr/bin/env python
"""
sunspot_area_from_sharp.py

Estimate sunspot (dark continuum) area in millionths of a solar hemisphere (MH)
for a given NOAA active region, using HMI SHARP CEA continuum data.

Workflow:
  1. Query a SHARP CEA series (default: hmi.sharp_720s_cea_nrt) over a time window.
  2. Find records whose NOAA_ARS list contains the requested NOAA AR.
  3. Take the latest such record.
  4. Download the corresponding CONTINUUM segment for that SHARP patch.
  5. Threshold the continuum image to isolate dark pixels (sunspots).
  6. Use the CEA pixel scale (CDELT1, CDELT2) to convert pixel count to MH.

Notes:
  * CEA projection is equal-area, so each pixel corresponds to the same area
    on the sphere. No additional foreshortening correction is needed.
  * Area in MH is computed as:
        area_MH = N_pixels * (dphi_rad * dy_rad) / (2*pi) * 1e6
    because a hemisphere has area 2*pi (in "radian^2" units on the unit sphere).
"""

import argparse
from datetime import datetime, timedelta, timezone
import re
import os

import numpy as np
import drms
from astropy.io import fits


try:
    from scipy import ndimage
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

JSOC_EMAIL = os.environ.get("JSOC_EMAIL", "your@email.here")

# -------------------- Utilities -------------------- #

def parse_iso_time(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_jsoc_tai_string(dt: datetime) -> str:
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.strftime("%Y.%m.%d_%H:%M:%S_TAI")


# -------------------- SHARP query -------------------- #

def find_latest_sharp_for_noaa(
    noaa_ar: int,
    center_time: datetime,
    window_hours: float = 48.0,
    series: str = "hmi.sharp_cea_720s_nrt",
    prefer_closest: bool = False,
):
    """
    Query JSOC for SHARP CEA records in a time window and find the latest one
    associated with the specified NOAA AR number.

    Returns a dict with:
      - harpnun
      - t_rec (JSOC string, e.g. '2024.07.31_12:00:00_TAI')
      - cdelt1_deg, cdelt2_deg (CEA pixel scale in degrees)
      - series
    Raises RuntimeError on failure.
    """
    client = drms.Client(email=JSOC_EMAIL)

    half = timedelta(hours=window_hours / 2.0)
    start = center_time - half
    end = center_time + half

    t_start = to_jsoc_tai_string(start)
    t_end = to_jsoc_tai_string(end)

    record_set = f"{series}[][ {t_start}-{t_end} ]"

    # Include CDELT1/2 so we don’t need them from the FITS header.
    keylist = ["HARPNUM", "T_REC", "NOAA_ARS", "CDELT1", "CDELT2"]

    print(f"[INFO] Querying JSOC series: {record_set}")
    df = client.query(record_set, key=keylist)

    if df.empty:
        raise RuntimeError(
            f"No SHARP records found for series {series} in window "
            f"{t_start} – {t_end}."
        )

    # Filter to rows whose NOAA_ARS list contains our AR number.
    noaa_str = str(noaa_ar)
    pattern = rf"(?:^|,){re.escape(noaa_str)}(?:,|$)"
    mask = df["NOAA_ARS"].astype(str).str.contains(pattern, regex=True)
    df_sel = df[mask]

    if df_sel.empty:
        raise RuntimeError(
            f"No SHARP CEA patches in {series} linked to NOAA AR {noaa_str} "
            f"in the window {t_start} – {t_end}."
        )

    # Parse times and either take "closest to center_time" or "latest",
    # depending on prefer_closest.
    df_sel = df_sel.copy()
    df_sel["T_REC_DT"] = drms.to_datetime(df_sel["T_REC"])
    df_sel = df_sel.dropna(subset=["T_REC_DT"])
    if df_sel.empty:
        raise RuntimeError("Matched records, but T_REC parsing failed.")

    if prefer_closest:
        # Make center_time naive UTC to match T_REC_DT
        if center_time.tzinfo is not None:
            center_naive = center_time.astimezone(timezone.utc).replace(tzinfo=None)
        else:
            center_naive = center_time

        df_sel["TIME_DELTA"] = (df_sel["T_REC_DT"] - center_naive).abs()
        latest = df_sel.sort_values("TIME_DELTA").iloc[0]
    else:
        # Original behavior: most recent record in the window
        latest = df_sel.sort_values("T_REC_DT").iloc[-1]

    harpnun = int(latest["HARPNUM"])
    t_rec = str(latest["T_REC"])
    cdelt1_deg = float(latest["CDELT1"])
    cdelt2_deg = float(latest["CDELT2"])

    # Full NOAA_ARS list for this SHARP (may contain multiple ARs)
    noaa_ars_str = str(latest["NOAA_ARS"]).strip()
    # Parse into a clean list of ints
    ar_tokens = re.split(r"[,\s]+", noaa_ars_str)
    noaa_ars_list = []
    for tok in ar_tokens:
        tok = tok.strip()
        if not tok:
            continue
        try:
            noaa_ars_list.append(int(tok))
        except ValueError:
            pass

    print(f"[INFO] Selected HARPNUM {harpnun} at {t_rec}")
    if noaa_ars_list:
        if len(noaa_ars_list) > 1:
            print(
                "[INFO] NOAA ARs in this SHARP patch: "
                + ", ".join(str(a) for a in noaa_ars_list)
            )
        else:
            print(f"[INFO] NOAA AR in this SHARP patch: {noaa_ars_list[0]}")
    else:
        print(f"[WARN] NOAA_ARS field empty or unparsable: {noaa_ars_str!r}")

    return {
        "harpnun": harpnun,
        "t_rec": t_rec,
        "cdelt1_deg": cdelt1_deg,
        "cdelt2_deg": cdelt2_deg,
        "series": series,
        "noaa_ars_str": noaa_ars_str,
        "noaa_ars_list": noaa_ars_list,
    }


# -------------------- Continuum download -------------------- #

def download_sharp_cont_and_mag(
    client,
    series: str,
    harpnun: int,
    t_rec: str,
    download_root: str,
):
    """
    Export and download the CONTINUUM and MAGNETOGRAM segments for a given
    SHARP CEA record.

    Returns (continuum_fits_path, magnetogram_fits_path).
    """
    # Ask JSOC for both segments in one export.
    # 'magnetogram' and 'continuum' are the standard segments for CEA SHARPs.
    rec = f"{series}[{harpnun}][{t_rec}]{{magnetogram,continuum}}"
    print(f"[INFO] Exporting continuum+magnetogram for record: {rec}")

    export = client.export(rec)


    # Directory to store output
    subdir_name = (
        f"{series.replace('.', '_')}"
        f"_H{harpnun}"
        f"_{t_rec.replace(':', '').replace('.', '')}"
    )
    tmpdir = os.path.join(download_root, subdir_name)
    os.makedirs(tmpdir, exist_ok=True)

    _ = export.download(tmpdir)

    fits_files = [
        os.path.join(tmpdir, f)
        for f in os.listdir(tmpdir)
        if f.lower().endswith(".fits")
    ]
    if not fits_files:
        raise RuntimeError(f"No .fits files found in {tmpdir} after export")

    cont_path = None
    mag_path = None
    for f in fits_files:
        name = os.path.basename(f).lower()
        if "continuum" in name:
            cont_path = f
        elif "magnetogram" in name:
            mag_path = f

    if cont_path is None or mag_path is None:
        raise RuntimeError(
            "Could not find both continuum and magnetogram FITS files "
            f"in {tmpdir} (found: {fits_files})"
        )

    print(f"[INFO] Downloaded continuum FITS:   {cont_path}")
    print(f"[INFO] Downloaded magnetogram FITS: {mag_path}")
    return cont_path, mag_path

# -------------------- Sunspot segmentation & area -------------------- #

def segment_sunspots(
    cont_data: np.ndarray,
    rel_thresh: float = 0.9,
    min_pixels: int = 10,
    use_local_bg: bool = True,
    bg_sigma: float = 25.0,
    pen_rel_thresh: float = 0.97,
    pen_radius: int = 5,
    qs_percentile: float = 50.0,
) -> np.ndarray:
    """
    Segment sunspots from HMI continuum.

    Steps:
      1. Optionally remove large-scale background (limb darkening) with a
         Gaussian-smoothed local background and normalize: norm = data / bg.
      2. Define:
           * core:      norm < rel_thresh      (umbra + darkest penumbra)
           * candidate: norm < pen_rel_thresh  (umbra + penumbra)
      3. Grow the core by `pen_radius` pixels and intersect with candidate
         to get penumbrae close to the core only.
      4. Remove blobs smaller than `min_pixels`.

    If SciPy is unavailable or use_local_bg=False, we fall back to using the
    raw intensities instead of the normalized image.
    """

    data = np.array(cont_data, copy=True)
    data = np.nan_to_num(data, nan=np.nanmedian(data))

    # Choose which field to threshold on: normalized or raw
    if use_local_bg and HAVE_SCIPY:
        bg = ndimage.gaussian_filter(data.astype(float), sigma=bg_sigma)

        good_bg = bg[bg > 0]
        if good_bg.size == 0:
            print(
                "[WARN] Background estimate invalid; "
                "falling back to global-median threshold on raw data."
            )
            field = data
        else:
            bg[bg <= 0] = np.median(good_bg)
            norm = data / bg
            field = norm
            print("[INFO] Using local background normalization.")
    else:
        field = data
        print("[INFO] Using raw continuum (no local background normalization).")


    # Quiet-Sun reference from the chosen field
    finite = field[np.isfinite(field)]
    if finite.size == 0:
        qs_level = 1.0
        print("[WARN] No finite pixels in field; using qs_level = 1.0")
    else:
        # Clip extremes to avoid very dark spots & very bright artifacts
        lo = np.percentile(finite, 1)
        hi = np.percentile(finite, 99)
        clipped = finite[(finite >= lo) & (finite <= hi)]
        if clipped.size == 0:
            clipped = finite
        qs_level = np.percentile(clipped, qs_percentile)
    print(
        f"[INFO] Quiet-Sun level p{qs_percentile:.1f} = {qs_level:.4f}"
    )

    core_thr = qs_level * rel_thresh
    pen_thr = qs_level * pen_rel_thresh
    print(
        f"[INFO] Thresholds: core<{core_thr:.4f} ({rel_thresh:.3f}×QS), "
        f"pen<{pen_thr:.4f} ({pen_rel_thresh:.3f}×QS)"
    )


    core = field < core_thr
    candidate = field < pen_thr

    # Penumbra limited to vicinity of the core
    if HAVE_SCIPY and pen_radius > 0:
        structure = ndimage.generate_binary_structure(2, 1)
        grown_core = ndimage.binary_dilation(core, structure=structure, iterations=pen_radius)
        penumbra = candidate & grown_core & ~core
        print(f"[INFO] Grown core by {pen_radius} pixels for penumbra.")
    else:
        penumbra = candidate & ~core

    mask = core | penumbra

    # Optional small-blob removal
    if min_pixels > 1 and HAVE_SCIPY:
        labeled, num = ndimage.label(mask)
        print(f"[INFO] Initial dark blobs: {num}")
        if num > 0:
            sizes = ndimage.sum(mask, labeled, index=np.arange(1, num + 1))
            remove_labels = np.where(sizes < min_pixels)[0] + 1
            for lab in remove_labels:
                mask[labeled == lab] = False
            print(
                f"[INFO] Removed {len(remove_labels)} blobs "
                f"smaller than {min_pixels} pixels."
            )
    elif min_pixels > 1 and not HAVE_SCIPY:
        print("[WARN] scipy not available; skipping small-blob removal.")

    return mask

def area_from_mask_in_mh(mask: np.ndarray, cdelt1_deg: float, cdelt2_deg: float) -> float:
    """
    Compute area in MH given a boolean mask and CEA pixel scales in degrees.

    For SHARP CEA (equal-area projection), each pixel covers:
        A_pix (unit sphere) = dphi_rad * dy_rad

    Hemispheric area (unit sphere) = 2*pi, so:

        area_MH = N * dphi_rad * dy_rad / (2*pi) * 1e6
    """
    n_pix = int(mask.sum())
    if n_pix == 0:
        return 0.0

    dphi_rad = np.deg2rad(abs(cdelt1_deg))
    dy_rad = np.deg2rad(abs(cdelt2_deg))

    area_mh = n_pix * dphi_rad * dy_rad / (2.0 * np.pi) * 1e6
    return area_mh

def save_output_plot(
    cont_data,
    mag_data,
    mask,
    fits_path,
    noaa_ar,
    t_rec,
    area_mh,
    out_dir=None,
    harpnun=None,
    noaa_ars_list=None,
    show_size=True,
):
    """
    Save output PNGs showing:
      TL: continuum
      TR: magnetogram
      BL: continuum + mask
      BR: magnetogram + mask
      plus the computed area in MH on the bottom row.
    """
    import matplotlib.pyplot as plt

    # Decide where to put the PNGs
    if out_dir is None:
        out_dir = os.path.dirname(fits_path)

    # Base name: AR####_<T_REC>
    safe_t = t_rec.replace(":", "").replace(".", "").replace("_TAI", "")
    base = f"AR{noaa_ar}_{safe_t}"

    # Continuum stretch
    cont = np.array(cont_data, copy=True)
    cont = np.nan_to_num(cont, nan=np.nanmedian(cont))
    c_vmin, c_vmax = np.percentile(cont, [1, 99])

    # Magnetogram stretch (symmetric around 0)
    mag = np.array(mag_data, copy=True)
    mag = np.nan_to_num(mag, nan=0.0)
    m_lo, m_hi = np.percentile(mag, [1, 99])
    m_abs = max(abs(m_lo), abs(m_hi))
    m_vmin, m_vmax = -m_abs, m_abs

    # Choose figure size adaptively based on continuum aspect ratio
    ny, nx = cont.shape
    aspect = nx / ny if ny > 0 else 2.0  # width / height of data

    # Base size for a "typical" AR patch
    base_w, base_h = 10.0, 7.5
    ref_aspect = 2.0

    # Scale width with aspect, but clamp so it doesn't get ridiculous
    scale = np.clip(aspect / ref_aspect, 1.0, 2.0)
    fig_w = base_w * scale
    fig_h = base_h

    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h))
    (ax_tl, ax_tr), (ax_bl, ax_br) = axes
    (ax_tl, ax_tr), (ax_bl, ax_br) = axes

    # Dark / night-mode background
    dark_bg = "#05070a"
    fig.patch.set_facecolor(dark_bg)
    for ax in axes.flat:
        ax.set_facecolor(dark_bg)

    # TL: continuum
    im_tl = ax_tl.imshow(
        cont,
        origin="lower",
        cmap="gray",
        vmin=c_vmin,
        vmax=c_vmax,
    )
    ax_tl.set_title("Continuum", color="#7bc2f9")

    # TR: magnetogram
    im_tr = ax_tr.imshow(
        mag,
        origin="lower",
        cmap="gray",
        vmin=m_vmin,
        vmax=m_vmax,
    )
    ax_tr.set_title("Magnetogram (LOS)", color="#7bc2f9")

    # BL: continuum + mask
    ax_bl.imshow(
        cont,
        origin="lower",
        cmap="gray",
        vmin=c_vmin,
        vmax=c_vmax,
    )
    overlay = np.zeros(mask.shape + (4,), dtype=float)
    overlay[mask] = [1.0, 0.0, 0.0, 0.3]
    ax_bl.imshow(overlay, origin="lower")
    ax_bl.set_title("Continuum + mask", color="#7bc2f9")

    # BR: magnetogram + mask
    ax_br.imshow(
        mag,
        origin="lower",
        cmap="gray",
        vmin=m_vmin,
        vmax=m_vmax,
    )
    ax_br.imshow(overlay, origin="lower")
    ax_br.set_title("Magnetogram + mask", color="#7bc2f9")

    # Area text on both bottom panels (optional)
    if show_size and area_mh is not None:
        text = f"Area ≈ {area_mh:.1f} MH"
        for ax in (ax_bl, ax_br):
            ax.text(
                0.02,
                0.95,
                text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                color="yellow",
                fontsize=10,
                bbox=dict(
                    facecolor="black",
                    alpha=0.6,
                    edgecolor="none",
                ),
            )

    # Remove all ticks / tick labels
    border_color = "#7bc2f9"
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(bottom=False, left=False)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(0.8)

    # --- Figure title + HARP/AR subtitle ---
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])

    fig.suptitle(
        f"{t_rec}",
        fontsize=14,
        y=0.97,
        color="#7bc2f9",
    )

    # Subtitle line with HARPNUM and all NOAA ARs in the SHARP patch
    if harpnun is not None:
        if noaa_ars_list:
            ar_label = ", ".join(f"AR {a}" for a in noaa_ars_list)
            subtitle = f"HARPNUM {harpnun} [{ar_label}]"
        else:
            subtitle = f"HARPNUM {harpnun}"

        fig.text(
            0.5,
            0.92,          
            subtitle,
            ha="center",
            va="center",
            color="#7bc2f9",
            fontsize=11,
        )

    panel_path = os.path.join(out_dir, f"{base}_4panel.png")
    fig.savefig(panel_path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print("[INFO] Saved 4-panel image:")
    print(f"  {panel_path}")

# -------------------- Main CLI -------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Estimate sunspot area in MH for a NOAA AR using SHARP CEA continuum."
    )
    parser.add_argument(
        "noaa_ar",
        type=int,
        help="NOAA active region number, e.g. 14294",
    )
    parser.add_argument(
        "--center-time", "--t",
        type=str,
        default=None,
        help="Center time for search window (ISO-like, e.g. 2024-07-31T12:00:00). "
             "Default: now (UTC).",
    )
    parser.add_argument(
        "--window-hours", "--w",
        type=float,
        default=48.0,
        help="Total width of search window in hours (default: 48).",
    )
    parser.add_argument(
        "--series",
        type=str,
        default="hmi.sharp_cea_720s_nrt",
        help=("SHARP CEA series to query. Examples:\n"
            "  hmi.sharp_cea_720s_nrt  (near real-time)\n"
            "  hmi.sharp_cea_720s      (definitive)\n"
            "Default: hmi.sharp_cea_720s_nrt"),
    )

    parser.add_argument(
        "--min-pixels",
        type=int,
        default=50,
        help="Minimum blob size in pixels to retain as sunspot (default: 30).",
    )
    parser.add_argument(
        "--no-size",
        action="store_true",
        help="Do not annotate plots with the computed area value.",
    )
    parser.add_argument(
        "--grow-pixels",
        type=int,
        default=0,
        help="Grow the final sunspot mask outward by this many pixels (binary dilation). Default: 0.",
    )
    parser.add_argument(
        "--no-local-bg",
        action="store_true",
        help="Disable local background normalization and use global median instead.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output PNGs (default: directory of the downloaded FITS).",
    )
    parser.add_argument(
        "--bmin",
        type=float,
        default=500.0,
        help="Minimum |B| (G) from magnetogram to accept a dark pixel as a spot "
             "(default: 150 G).",
    )
    parser.add_argument(
        "--rel-thresh",
        type=float,
        default=0.8,
        help="Relative threshold vs quiet-Sun median for dark pixels (default: 0.9).",
    )
    parser.add_argument(
        "--pen-rel-thresh",
        type=float,
        default=.985,
        help="Relative threshold vs quiet-Sun for candidate penumbra pixels "
             "(default: 0.97).",
    )
    parser.add_argument(
        "--pen-radius",
        type=int,
        default=200,
        help="Max distance in pixels from core spots to include as penumbra "
             "(default: 5).",
    )
    parser.add_argument(
        "--qs-percentile",
        type=float,
        default=33,
        help=(
            "Percentile of continuum intensities to treat as quiet-Sun level "
            "(default 50 = median). Try 60–70 for big limb regions."
        ),
    )
    parser.add_argument(
        "--bg-sigma",
        type=float,
        default=70.0,
        help="Gaussian sigma in pixels for local background (default: 25).",
    )
    parser.add_argument(
        "--ignore-x-left",
        type=int,
        default=0,
        help="Number of leftmost CEA columns to ignore (force mask=False).",
    )
    parser.add_argument(
        "--ignore-x-right",
        type=int,
        default=0,
        help="Number of rightmost CEA columns to ignore.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="sharp_data",
        help=(
            "Root directory for downloaded SHARP FITS and PNGs "
            "(default: ./sharp_data)."
        ),
    )


    args = parser.parse_args()

    root_dir = os.path.abspath(args.data_dir)
    ar_dir = os.path.join(root_dir, f"AR{args.noaa_ar}")
    os.makedirs(ar_dir, exist_ok=True)

    if args.center_time is None:
        center_time = datetime.now(timezone.utc)
        prefer_closest = False   # realtime mode → use "latest in window"
    else:
        center_time = parse_iso_time(args.center_time)
        prefer_closest = True    # user specified a time → use "closest"

    try:
        meta = find_latest_sharp_for_noaa(
            noaa_ar=args.noaa_ar,
            center_time=center_time,
            window_hours=args.window_hours,
            series=args.series,
            prefer_closest=prefer_closest,
        )

    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return

    # If this SHARP contains multiple NOAA ARs, note that explicitly.
    ar_list = meta.get("noaa_ars_list", [])
    if ar_list and len(ar_list) > 1:
        others = [a for a in ar_list if a != args.noaa_ar]
        if others:
            print(
                "[NOTE] Requested AR "
                f"{args.noaa_ar} is part of a multi-AR SHARP patch.\n"
                "       This HARP also contains: "
                + ", ".join(str(a) for a in others)
                + " (area estimate will include all of them)."
            )
        else:
            # edge case: duplicates / only the requested AR
            print(
                "[NOTE] SHARP patch lists multiple NOAA_ARS entries, "
                "but they all match the requested AR."
            )

    client = drms.Client(email=JSOC_EMAIL)

    try:
        cont_fits_path, mag_fits_path = download_sharp_cont_and_mag(
            client,
            series=meta["series"],
            harpnun=meta["harpnun"],
            t_rec=meta["t_rec"],
            download_root=ar_dir,
        )
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return

    print(f"[INFO] Loaded continuum FITS:   {cont_fits_path}")
    print(f"[INFO] Loaded magnetogram FITS: {mag_fits_path}")

    with fits.open(cont_fits_path) as hdul:
        hdul.verify("silentfix+warn")
        cont_data = hdul[1].data

    with fits.open(mag_fits_path) as hdul:
        hdul.verify("silentfix+warn")
        mag_data = hdul[1].data

    mask = segment_sunspots(
        cont_data,
        rel_thresh=args.rel_thresh,
        min_pixels=args.min_pixels,
        use_local_bg=not args.no_local_bg,
        bg_sigma=args.bg_sigma,
        pen_rel_thresh=args.pen_rel_thresh,
        pen_radius=args.pen_radius,
        qs_percentile=args.qs_percentile,
    )

    if args.grow_pixels > 0 and HAVE_SCIPY:
        structure = ndimage.iterate_structure(
            ndimage.generate_binary_structure(2, 1),
            args.grow_pixels,
        )
        mask = ndimage.binary_dilation(mask, structure=structure)
        print(f"[INFO] Grown mask outward by ~{args.grow_pixels} pixels.")
    elif args.grow_pixels > 0 and not HAVE_SCIPY:
        print("[WARN] --grow-pixels requested but SciPy is not available; skipping dilation.")
        
    # Magnetic gating: keep only dark pixels where |B| >= bmin
    # Magnetic gating: keep only blobs that contain sufficiently strong |B|
    if mag_data is not None and args.bmin > 0:
        mag_abs = np.abs(np.nan_to_num(mag_data, nan=0.0))
        before = int(mask.sum())

        if HAVE_SCIPY:
            labeled, num = ndimage.label(mask)
            keep_labels = []

            for lab in range(1, num + 1):
                comp_idx = (labeled == lab)
                if not np.any(comp_idx):
                    continue
                comp_b = mag_abs[comp_idx]
                # Use max or a high percentile as "field strength" of this blob
                if np.nanmax(comp_b) >= args.bmin:
                    keep_labels.append(lab)

            if keep_labels:
                keep_labels = np.array(keep_labels, dtype=int)
                # build new mask that keeps only selected labels
                mask = np.isin(labeled, keep_labels)
            else:
                mask[:] = False

            after = int(mask.sum())
            print(
                f"[INFO] Applied blob-wise |B| >= {args.bmin:.1f} G filter: "
                f"{before} -> {after} spot pixels across {num} blobs."
            )
        else:
            # Fallback: old pixel-wise gating if SciPy isn't available
            field_mask = mag_abs >= args.bmin
            mask &= field_mask
            after = int(mask.sum())
            print(
                f"[INFO] Applied pixel-wise |B| >= {args.bmin:.1f} G filter: "
                f"{before} -> {after} spot pixels."
            )


    # Optional hard clipping of edge columns (for nasty limb wedges)
    if args.ignore_x_left > 0:
        mask[:, :args.ignore_x_left] = False
        print(f"[INFO] Ignored leftmost {args.ignore_x_left} columns in mask.")
    if args.ignore_x_right > 0:
        mask[:, -args.ignore_x_right:] = False
        print(f"[INFO] Ignored rightmost {args.ignore_x_right} columns in mask.")

    n_spot_pixels = int(mask.sum())
    area_mh = area_from_mask_in_mh(
        mask,
        cdelt1_deg=meta["cdelt1_deg"],
        cdelt2_deg=meta["cdelt2_deg"],
    )

    # Choose where plots go; default to the AR dir
    plot_out_dir = args.output_dir if args.output_dir is not None else ar_dir

    save_output_plot(
        cont_data=cont_data,
        mag_data=mag_data,
        mask=mask,
        fits_path=cont_fits_path,
        noaa_ar=args.noaa_ar,
        t_rec=meta["t_rec"],
        area_mh=area_mh,
        out_dir=plot_out_dir,
        harpnun=meta.get("harpnun"),
        noaa_ars_list=meta.get("noaa_ars_list"),
        show_size=not args.no_size,
    )

    print()
    print("=== Sunspot Area Estimate (from SHARP CEA continuum) ===")
    print(f"NOAA AR:         {args.noaa_ar}")
    print(f"Series:          {meta['series']}")
    print(f"HARPNUM:         {meta['harpnun']}")
    print(f"T_REC:           {meta['t_rec']}")
    noaa_ars_str = meta.get("noaa_ars_str", "")
    if noaa_ars_str:
        print(f"NOAA ARs in SHARP: {noaa_ars_str}")
    print(f"CDELT1,2 (deg):  {meta['cdelt1_deg']:.5f}, {meta['cdelt2_deg']:.5f}")
    print(f"Spot pixels:     {n_spot_pixels}")
    print(f"Estimated area:  {area_mh:.1f} MH (millionths of a hemisphere)")
    if n_spot_pixels == 0:
        print("WARNING: no dark pixels found; consider raising rel_thresh.")
    if n_spot_pixels == 0:
        print("WARNING: no dark pixels found; consider raising rel_thresh.")
    print()


if __name__ == "__main__":
    main()
