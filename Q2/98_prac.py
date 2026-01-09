import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import radon

def circular_fov_mask(img: np.ndarray):
    """원형 FOV 밖을 0으로 만드는 마스크(CT 팬텀/시야 제한용)."""
    h, w = img.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 2

    y, x = np.ogrid[:h, :w]
    outside = (x - cx) ** 2 + (y - cy) ** 2 > r ** 2

    out = img.copy()
    out[outside] = 0
    return out

def show_sinogram_like_example(
    image_path: str,
    pixel_size_mm: float = 1.0,
    n_angles: int | None = None,
    apply_circle_mask: bool = True,
):
    img_u8 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_u8 is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
    
    img = img_u8.astype(np.float32)

    if apply_circle_mask:
        img = circular_fov_mask(img)

    if n_angles is None:
        theta = np.linspace(0.0, 180.0, 180, endpoint=False)
    else:
        theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)

    sino = radon(img, theta=theta, circle=False)

    sino_plot = sino.T

    h, w = img.shape
    x_extent_mm = (-(w / 2) * pixel_size_mm, (w / 2) * pixel_size_mm)
    y_extent_mm = (-(h / 2) * pixel_size_mm, (h / 2) * pixel_size_mm)

    num_r = sino.shape[0]
    r_max_mm = (num_r / 2) * pixel_size_mm
    r_extent_mm = (-r_max_mm, r_max_mm)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    ax1.set_title("Original")
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")
    ax1.imshow(
        img,
        cmap="gray",
        extent=(x_extent_mm[0], x_extent_mm[1], y_extent_mm[0], y_extent_mm[1]),
        interpolation="nearest",
    )

    ax2.set_title("Radon transform (Sinogram)")
    ax2.set_xlabel("r [mm]")
    ax2.set_ylabel(r"$\phi$ [degree]")
    ax2.imshow(
        sino_plot,
        cmap="gray",
        extent=(r_extent_mm[0], r_extent_mm[1], theta[0], theta[-1]),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_sinogram_like_example(
        image_path=r"C:\Users\HEECHEOL\Desktop\grad\05_DL\02_DL\Q2\origin_circle.png",
        pixel_size_mm=1.0,
        n_angles=180,
        apply_circle_mask=True,
    )