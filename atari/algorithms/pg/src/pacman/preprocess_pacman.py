import numpy as np
import cv2


def preprocess_pacman_frame(image_frame: np.ndarray) -> np.ndarray:
    """
    Preprocess MsPacman frame to preserve important color information.
    - Resize to 80x80
    - Keep color channels but normalize
    - Apply contrast enhancement
    """
    # Resize to 80x80 while preserving color
    resized = cv2.resize(image_frame, (80, 80), interpolation=cv2.INTER_AREA)

    # Convert to LAB color space for better contrast enhancement
    lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)

    # Apply CLAHE to L channel only
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # Convert back to RGB
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Normalize to [0, 1]
    norm = enhanced.astype(np.float32) / 255.0

    return norm.ravel()


def preprocess_pacman_frame_color_aware(image_frame: np.ndarray) -> np.ndarray:
    """
    Color-aware preprocessing that extracts specific color features for MsPacman.
    - Extract ghost colors (red, pink, blue, orange)
    - Extract power pellet colors
    - Extract Pacman color
    - Create feature channels for each important color
    """
    # Resize to 80x80
    resized = cv2.resize(image_frame, (80, 80), interpolation=cv2.INTER_AREA)

    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)

    # Define color ranges for important game elements
    # Red ghost (hue around 0 or 180)
    red_mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    red_mask = red_mask1 | red_mask2

    # Pink ghost (hue around 150-170)
    pink_mask = cv2.inRange(hsv, np.array([150, 100, 100]), np.array([170, 255, 255]))

    # Blue ghost (hue around 100-120)
    blue_mask = cv2.inRange(hsv, np.array([100, 100, 100]), np.array([120, 255, 255]))

    # Orange ghost (hue around 10-25)
    orange_mask = cv2.inRange(hsv, np.array([10, 100, 100]), np.array([25, 255, 255]))

    # Yellow Pacman (hue around 20-30)
    yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))

    # White dots (high value, low saturation)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))

    # Convert masks to float and normalize
    features = []
    for mask in [red_mask, pink_mask, blue_mask, orange_mask, yellow_mask, white_mask]:
        features.append(mask.astype(np.float32) / 255.0)

    # Add grayscale as additional feature
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    gray_norm = gray.astype(np.float32) / 255.0
    features.append(gray_norm)

    # Stack all features
    feature_stack = np.stack(features, axis=-1)  # Shape: (80, 80, 7)

    return feature_stack.ravel()  # Flatten to 1D


def preprocess_pacman_frame_difference(
    current_frame: np.ndarray, previous_frame: np.ndarray = None
) -> np.ndarray:
    """
    Preprocess frame with differencing to detect motion.
    This is often more effective for Atari games.
    """
    current_processed = preprocess_pacman_frame(current_frame)

    if previous_frame is None:
        # If no previous frame, return zeros
        return np.zeros_like(current_processed)

    previous_processed = preprocess_pacman_frame(previous_frame)

    # Compute frame difference
    frame_diff = current_processed - previous_processed

    # Normalize the difference
    if np.std(frame_diff) > 1e-8:
        frame_diff = (frame_diff - np.mean(frame_diff)) / np.std(frame_diff)

    return frame_diff


def preprocess_pacman_frame_color_aware_difference(
    current_frame: np.ndarray, previous_frame: np.ndarray = None
) -> np.ndarray:
    """
    Color-aware preprocessing with frame differencing.
    """
    current_processed = preprocess_pacman_frame_color_aware(current_frame)

    if previous_frame is None:
        # If no previous frame, return zeros
        return np.zeros_like(current_processed)

    previous_processed = preprocess_pacman_frame_color_aware(previous_frame)

    # Compute frame difference
    frame_diff = current_processed - previous_processed

    # Normalize the difference
    if np.std(frame_diff) > 1e-8:
        frame_diff = (frame_diff - np.mean(frame_diff)) / np.std(frame_diff)

    return frame_diff
