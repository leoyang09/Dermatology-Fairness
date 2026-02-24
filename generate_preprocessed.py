import cv2
import numpy as np
import os
import glob
from multiprocessing import Pool, cpu_count
import tqdm
import argparse

def apply_clahe(img, clip_limit=2.0):
    """Apply CLAHE to L channel of LAB image."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def apply_adaptive_gamma(img):
    """Apply Adaptive Gamma Correction."""
    
    # Convert to HSV to get Value channel (brightness)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Compute mean brightness
    m = np.mean(v) / 255.0

    # Avoid division by zero or log of zero
    m = np.clip(m, 1e-5, 1.0)

    # Calculate gamma: mapping mean brightness to 0.5
    # log(0.5) / log(m)
    # If m is 0.5, gamma is 1.
    # If m < 0.5 (dark), gamma < 1 (brighten).
    # If m > 0.5 (bright), gamma > 1 (darken).
    gamma = np.log(0.5) / np.log(m)

    # Apply gamma correction to V channel
    # Standard: I_out = I_in ^ gamma
    table = np.array(
        [((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")

    v_corrected = cv2.LUT(v, table)

    hsv_corrected = cv2.merge((h, s, v_corrected))

    return cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR)
    
def apply_white_balance(img):
    """Apply Gray World White Balance (LAB-based)."""
    
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])

    result[:, :, 1] = result[:, :, 1] - (
        (avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1
    )
    result[:, :, 2] = result[:, :, 2] - (
        (avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1
    )

    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    return result


def apply_white_balance_simple(img):
    """Simple Gray World: Scale channels so means are equal."""
    
    b, g, r = cv2.split(img)

    m_b = np.mean(b)
    m_g = np.mean(g)
    m_r = np.mean(r)

    avg = (m_b + m_g + m_r) / 3

    # Scale factors
    k_b = avg / m_b if m_b > 0 else 1
    k_g = avg / m_g if m_g > 0 else 1
    k_r = avg / m_r if m_r > 0 else 1

    b = cv2.convertScaleAbs(b, alpha=k_b)
    g = cv2.convertScaleAbs(g, alpha=k_g)
    r = cv2.convertScaleAbs(r, alpha=k_r)

    return cv2.merge((b, g, r))


def apply_clahe_adaptive_gamma(img):
    """Apply CLAHE -> Adaptive Gamma."""
    img = apply_clahe(img)
    img = apply_adaptive_gamma(img)
    return img

# -------------------------
# New preprocessing methods
# -------------------------

def apply_msrcr(
    img: np.ndarray,
    gamma=128,
    scales=(15, 80, 250),
    alpha=125.0,
    beta=46.0,
    gain=1.0,
    offset=0.0
) -> np.ndarray:
    """
    MSRCR (Multi-Scale Retinex with Color Restoration).
    - Retinex: sum_i (log(I) - log(blur(I, scale_i)))
    - Color restoration: beta * (log(alpha*I) - log(sum_channels(I)))
    Notes:
    - This is a common practical MSRCR variant. Parameters can be tuned.
    """
    gain = gamma / 128.0
    img_f = img.astype(np.float32) + 1.0  # avoid log(0)
    retinex = np.zeros_like(img_f)

    for s in scales:
        blur = cv2.GaussianBlur(img_f, (0, 0), sigmaX=float(s), sigmaY=float(s))
        retinex += np.log(img_f) - np.log(blur + 1.0)

    retinex /= float(len(scales))

    # Color restoration
    sum_channels = np.sum(img_f, axis=2, keepdims=True)
    color_restoration = beta * (np.log(alpha * img_f) - np.log(sum_channels + 1.0))

    msrcr = gain * (retinex * color_restoration) + offset

    # Per-channel normalization to 0..255
    out = np.zeros_like(msrcr)
    for c in range(3):
        ch = msrcr[:, :, c]
        ch = ch - np.min(ch)
        denom = np.max(ch) + 1e-6
        ch = (ch / denom) * 255.0
        out[:, :, c] = ch

    return np.clip(out, 0, 255).astype(np.uint8)


def apply_homomorphic(
    img: np.ndarray,
    cutoff=0.35,
    order=2.0,
    boost=1.6
) -> np.ndarray:
    """
    Homomorphic filtering for illumination normalization (on V channel).
    Steps:
    - work in log domain
    - FFT
    - apply high-pass filter (Butterworth-like radial)
    - inverse FFT
    - exp + normalize
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    v = np.clip(v, 1.0, 255.0)
    v_log = np.log(v)

    # FFT
    v_fft = np.fft.fft2(v_log)
    v_fft_shift = np.fft.fftshift(v_fft)

    rows, cols = v.shape
    cy, cx = rows // 2, cols // 2
    y = np.arange(rows) - cy
    x = np.arange(cols) - cx
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X * X + Y * Y)

    D0 = cutoff * np.max(D)  # cutoff as fraction of max radius
    D0 = max(D0, 1e-6)

    # High-pass Butterworth-like
    H = 1.0 - 1.0 / (1.0 + (D / D0) ** (2.0 * order))

    # Apply filter with boost
    v_filt = v_fft_shift * (1.0 + (boost - 1.0) * H)

    # Inverse FFT
    v_ishift = np.fft.ifftshift(v_filt)
    v_ifft = np.fft.ifft2(v_ishift)
    v_out = np.real(v_ifft)

    # Back from log domain
    v_out = np.exp(v_out)

    # Normalize to 0..255
    v_out = v_out - np.min(v_out)
    v_out = v_out / (np.max(v_out) + 1e-6)
    v_out = (v_out * 255.0).astype(np.uint8)

    hsv_out = cv2.merge((h.astype(np.uint8), s.astype(np.uint8), v_out))
    return cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)


def apply_percentile_norm(
    img: np.ndarray,
    range_=98.0
) -> np.ndarray:
    """
    Robust brightness normalization using percentiles (on V channel).
    Clips and stretches brightness so [low_p, high_p] maps to [0,255].
    """
    low_p = (100.0 - range_) / 2.0
    high_p = 100.0 - low_p
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v_f = v.astype(np.float32)
    lo = np.percentile(v_f, low_p)
    hi = np.percentile(v_f, high_p)

    if hi <= lo + 1e-6:
        return img

    v_f = (v_f - lo) / (hi - lo)
    v_f = np.clip(v_f, 0.0, 1.0) * 255.0
    v_out = v_f.astype(np.uint8)

    hsv_out = cv2.merge((h, s, v_out))
    return cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)


def apply_local_contrast(
    img: np.ndarray,
    sigma=2.0,
    strength=1.25
) -> np.ndarray:
    """
    Local feature enhancement via unsharp masking on luminance (L in LAB).
    - strength > 1 boosts local contrast
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)

    blur = cv2.GaussianBlur(l, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    detail = l - blur
    l_enh = l + strength * detail

    l_enh = np.clip(l_enh, 0, 255)
    lab_out = cv2.merge((l_enh, a, b)).astype(np.uint8)
    return cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)


def apply_erythema_enhance(
    img: np.ndarray,
    strength=0.6,
    softness=6.0
) -> np.ndarray:
    """
    Medical-ish erythema emphasis.
    Computes an erythema proxy index and boosts redness in high-index areas.

    Proxy index (simple, fast):
      EI = 2R - G - B
    Then:
      mask = sigmoid(normalized_EI)
      R += strength * mask * (255 - R)
      G,B slightly decreased in those regions

    This is not a diagnostic tool. It's a visualization-style enhancement.
    """
    bgr = img.astype(np.float32)
    b, g, r = cv2.split(bgr)

    ei = 2.0 * r - g - b  # erythema proxy
    # Normalize EI to 0..1 robustly
    ei_lo = np.percentile(ei, 5.0)
    ei_hi = np.percentile(ei, 95.0)
    if ei_hi <= ei_lo + 1e-6:
        return img

    ei_n = (ei - ei_lo) / (ei_hi - ei_lo)
    ei_n = np.clip(ei_n, 0.0, 1.0)

    # Sigmoid mask for smooth emphasis
    mask = 1.0 / (1.0 + np.exp(-softness * (ei_n - 0.5)))

    # Apply adjustments
    r_out = r + strength * mask * (255.0 - r)
    g_out = g * (1.0 - 0.15 * strength * mask)
    b_out = b * (1.0 - 0.15 * strength * mask)

    out = cv2.merge((b_out, g_out, r_out))
    return np.clip(out, 0, 255).astype(np.uint8)

def apply_illumination_comp(
    img: np.ndarray,
    sigma=50.0,
    eps=1e-6
) -> np.ndarray:
    """
    Illumination compensation via large-scale Gaussian illumination estimation.
    Steps:
    - Convert to LAB
    - Estimate illumination via heavy Gaussian blur on L channel
    - Divide original L by illumination
    - Normalize back to 0..255
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    # Estimate illumination field
    illum = cv2.GaussianBlur(l, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    # Avoid divide-by-zero
    illum = illum + eps
    # Remove illumination
    l_corr = l / illum
    l_corr = l_corr - np.min(l_corr)
    l_corr = l_corr / (np.max(l_corr) + eps)
    l_corr = l_corr * 255.0
    lab_out = cv2.merge((l_corr, a, b)).astype(np.uint8)
    return cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

def apply_z_score_norm(
    img: np.ndarray,
    scale=1.0,
    eps=1e-6
) -> np.ndarray:
    """
    Per-channel Z-score normalization.
    - Subtract mean
    - Divide by std (adjusted by scale)
    - Rescale to 0..255 for image saving
    """

    img_f = img.astype(np.float32)

    out = np.zeros_like(img_f)

    for c in range(3):
        channel = img_f[:, :, c]
        mean = np.mean(channel)
        std = np.std(channel)

        std = max(std * scale, eps)

        z = (channel - mean) / std

        # Normalize to 0..255 for saving
        z = z - np.min(z)
        z = z / (np.max(z) + eps)
        z = z * 255.0

        out[:, :, c] = z

    return np.clip(out, 0, 255).astype(np.uint8)

# -------------------------
# Additional New Methods
# -------------------------

def apply_color_constancy(img, power=6, sigma=0):
    """
    Shades of Gray color constancy algorithm.
    Generalization of gray world and white patch.

    power=1  -> gray world
    power=inf -> white patch

    Optionally applies Gaussian smoothing before computing
    the Minkowski norm.
    """
    img_float = img.astype(np.float32) + 1e-6

    # Optional smoothing
    if sigma > 0:
        img_float = cv2.GaussianBlur(img_float, (0, 0), sigma)

    # Minkowski norm across spatial dimensions
    norm = np.power(np.mean(np.power(img_float, power), axis=(0, 1)), 1/power)

    # Normalize each channel
    result = img.astype(np.float32)
    for i in range(3):
        if norm[i] > 0:
            result[:, :, i] = result[:, :, i] / norm[i] * np.mean(norm)

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_lab_color_normalization(img, target_a=128, target_b=128):
    """
    Normalize a and b channels in LAB color space.

    Shifts the mean of the chromatic channels (a and b)
    to a specified target value (default centers).
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Shift a and b channels to target mean
    lab[:, :, 1] = lab[:, :, 1] - np.mean(lab[:, :, 1]) + target_a
    lab[:, :, 2] = lab[:, :, 2] - np.mean(lab[:, :, 2]) + target_b

    lab = np.clip(lab, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def apply_clahe_mild(img):
    """
    CLAHE with lower clip limit (1.0).
    Provides gentle contrast enhancement.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_clahe_very_mild(img):
    """
    Very mild CLAHE (clip limit 0.5).
    Minimal contrast adjustment.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_clahe_blended(img, strength=0.5):
    """
    Apply partial CLAHE and blend with original.

    strength=0.5 means:
    50% enhanced image + 50% original image.
    """
    enhanced = apply_clahe(img)
    blended = cv2.addWeighted(img, 1 - strength, enhanced, strength, 0)
    return blended


def apply_white_balance_mild(img, strength=0.3):
    """
    Apply partial white balance and blend with original.

    strength=0.3 means:
    30% white-balanced image + 70% original.
    """
    wb = apply_white_balance_simple(img)
    blended = cv2.addWeighted(img, 1 - strength, wb, strength, 0)
    return blended


def apply_white_balance_very_mild(img, strength=0.15):
    """
    Very subtle white balance correction.

    strength=0.15 means:
    15% corrected image + 85% original.
    """
    wb = apply_white_balance_simple(img)
    blended = cv2.addWeighted(img, 1 - strength, wb, strength, 0)
    return blended


def apply_bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
    """
    Edge-preserving noise reduction.

    Smooths homogeneous regions while maintaining sharp edges.
    """
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def apply_non_local_means(img, h=10, template_window=7, search_window=21):
    """
    Non-local means denoising.

    State-of-the-art noise reduction that averages
    similar patches across the image.
    """
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, template_window, search_window)

def apply_skin_tone_shift(img, shift_amount=20):
    """
    Shift skin tone by adjusting color channels.
    Simulates how same lesion might look on different skin.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Shift hue (skin tone)
    hsv[:, :, 0] = (hsv[:, :, 0] + shift_amount) % 180
    
    # Optionally adjust value (brightness)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.9, 0, 255)
    
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

# -------------------------
# Define pairs (DeepDerm + HAM10000)
# -------------------------
pairs = [
]

triples = [

    # DeepDerm strongest theoretical stack
    ("non_local_means", "msrcr", "adaptive_gamma"),

    # Balanced strong fairness stack
    ("bilateral", "msrcr", "local_contrast"),

    # HAM10000-friendly stack
    ("non_local_means", "illumination_comp", "clahe"),

    # Statistical + Retinex hybrid
    ("bilateral", "msrcr", "Z_score_norm"),

    # Conservative fairness pipeline
    ("non_local_means", "illumination_comp", "percentile_norm"),
]

# Map method names to functions
method_map = {
    "clahe": apply_clahe,
    "adaptive_gamma": apply_adaptive_gamma,
    "white_balance": apply_white_balance,
    "msrcr": apply_msrcr,
    "homomorphic": apply_homomorphic,
    "percentile_norm": apply_percentile_norm,
    "local_contrast": apply_local_contrast,
    "illumination_comp": apply_illumination_comp,
    "Z_score_norm": apply_z_score_norm,
    "color_constancy": apply_color_constancy,
    "lab_color_norm": apply_lab_color_normalization,
    "clahe_mild": apply_clahe_mild,
    "clahe_very_mild": apply_clahe_very_mild,
    "clahe_blended": apply_clahe_blended,
    "white_balance_mild": apply_white_balance_mild,
    "white_balance_very_mild": apply_white_balance_very_mild,
    "bilateral": apply_bilateral_filter,
    "non_local_means": apply_non_local_means,
    "skin_tone_shift": apply_skin_tone_shift,
}
# -------------------------
# Multiprocessing worker
# -------------------------

def process_image(args):
    if len(args) == 3:
        img_path, out_dir, func1 = args
        filename = os.path.basename(img_path)
        try:
            img = cv2.imread(img_path)
            if img is None:
                return f"Failed to read {img_path}"
            img = func1(img)
            cv2.imwrite(os.path.join(out_dir, filename), img)
            return None
        except Exception as e:
            return f"{filename}: {str(e)}"

    img_path, output_dirs = args
    filename = os.path.basename(img_path)

    try:
        img = cv2.imread(img_path)
        if img is None:
            return f"Failed to read {img_path}"

        # New preprocessing methods only

        return None
    except Exception as e:
        return f"Error processing {filename}: {str(e)}"
def process_image_pair(args):
    img_path, out_dir, func1, func2 = args
    filename = os.path.basename(img_path)
    try:
        img = cv2.imread(img_path)
        if img is None:
            return f"Failed to read {img_path}"
        img = func1(img)
        img = func2(img)
        cv2.imwrite(os.path.join(out_dir, filename), img)
        return None
    except Exception as e:
        return f"{filename}: {str(e)}"

def process_image_triple(args):
    img_path, out_dir, func1, func2, func3 = args
    filename = os.path.basename(img_path)

    try:
        img = cv2.imread(img_path)
        if img is None:
            return f"Failed to read {img_path}"

        img = func1(img)
        img = func2(img)
        img = func3(img)

        cv2.imwrite(os.path.join(out_dir, filename), img)
        return None

    except Exception as e:
        return f"{filename}: {str(e)}"
# -------------------------
# CLI entry
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="DDI", help="Root DDI directory")
    parser.add_argument("--exts", default="png,jpg,jpeg", help="Comma-separated image extensions")
    args = parser.parse_args()

    img_dir = os.path.join(args.data_dir, "images")
    if not os.path.exists(img_dir):
        print(f"Error: {img_dir} does not exist.")
        return

    # Define single-method output directories
    output_dirs = {
        
    }

    for d in output_dirs.values():
        os.makedirs(d, exist_ok=True)


    # Find images
    print("Finding images...")
    exts = [e.strip().lower() for e in args.exts.split(",") if e.strip()]
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(img_dir, f"*.{ext}")))
    image_paths = sorted(set(image_paths))

    print(f"Found {len(image_paths)} images.")
    if len(image_paths) == 0:
        print("No images found. Exiting.")
        return

    # Prepare arguments for multiprocessing
    tasks = [(p, output_dirs) for p in image_paths]

    n_cores = cpu_count()
    print(f"Processing images using {n_cores} cores...")

    with Pool(processes=n_cores) as p:
        results = list(tqdm.tqdm(p.imap(process_image, tasks), total=len(tasks)))

    # Report errors if any
    errors = [r for r in results if isinstance(r, str) and r]
    if errors:
        print("\nSome images failed:")
        for e in errors[:50]:
            print(e)
        if len(errors) > 50:
            print(f"... and {len(errors) - 50} more.")

    # -------------------------
    # Run all pairs
    # -------------------------
    for m1, m2 in pairs:
        out_dir = os.path.join(args.data_dir, f"images_{m1}_{m2}")
        os.makedirs(out_dir)
        print(f"\nGenerating pair: {m1} -> {m2}")

        func1 = method_map[m1]
        func2 = method_map[m2]
        tasks = [(p, out_dir, func1, func2) for p in image_paths]

        with Pool(processes=n_cores) as p:
            results = list(tqdm.tqdm(p.imap(process_image_pair, tasks), total=len(tasks)))

        errors = [r for r in results if isinstance(r, str) and r]
        if errors:
            print(f"{len(errors)} images failed for pair {m1}->{m2}")
            for e in errors[:50]:
                print(e)


    # -------------------------
    # Run all triples
    # -------------------------
    for m1, m2, m3 in triples:

        out_dir = os.path.join(
            args.data_dir,
            f"images_{m1}_{m2}_{m3}"
        )
        os.makedirs(out_dir, exist_ok=True)

        print(f"\nGenerating triple: {m1} -> {m2} -> {m3}")

        func1 = method_map[m1]
        func2 = method_map[m2]
        func3 = method_map[m3]

        tasks = [
            (p, out_dir, func1, func2, func3)
            for p in image_paths
        ]

        with Pool(processes=n_cores) as pool:
            results = list(tqdm.tqdm(
                pool.imap(process_image_triple, tasks),
                total=len(tasks)
            ))

        errors = [r for r in results if isinstance(r, str) and r]
        if errors:
            print(f"{len(errors)} images failed for triple {m1}->{m2}->{m3}")
            for e in errors[:50]:
                print(e)

    print("Processing completed.")

if __name__ == "__main__":
    main()