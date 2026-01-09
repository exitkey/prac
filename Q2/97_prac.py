import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import radon

# 원형 FOV 밖을 0으로 만드는 마스크
# def circular_fov_mask(img: np.ndarray):
#     h, w = img.shape
#     cy, cx = h // 2, w // 2
#     r = min(h, w) // 2

#     y, x = np.ogrid[:h, :w] # 메모리 덜 쓰는 좌표 격자 생성
#     outside = (x - cx) ** 2 + (y - cy) ** 2 > r ** 2

#     out = img.copy() # 복사
#     out[outside] = 0 
#     return out

def show_sinogram_like_example(
    image_path: str,
    pixel_size_mm: float = 1.0,   # 1픽셀이 몇 mm인지, 여기만 바꾸면 r축(mm) 스케일이 맞음
    n_angles: int | None = None,  # None이면 180(0~179도) 사용
   # apply_circle_mask: bool = True, # True면 원형 마스크 적용
):
    
    # 1) 이미지 로드 (grayscal)
    img_u8 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_u8 is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}") # raise: 예외 함수

    # 2) float 변환 (radon에 넣기 편하게)
    img = img_u8.astype(np.float32) # radon은 float 입력이 더 자연스러움(소수점), uint8은 연산 중 오버플로/정밀도 이슈가 생길 수 있음

    # 3) 원형 FOV 마스크(선택)
    # if apply_circle_mask:
    #     img = circular_fov_mask(img)

    # 4) 각도(theta) 설정
    if n_angles is None:
        # 너가 원한 오른쪽 그림 느낌(0~180 미만, 1도 간격)
        theta = np.linspace(0.0, 180.0, 180, endpoint=False) # 0부터 180까지 180개의 균등 샘플
    else:
        theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)

    # 5) Radon transform -> sinogram
    # circle=False: 우리가 직접 마스크를 적용했으니 False로 둬도 OK
    sino = radon(img, theta=theta, circle=False)  # shape: (num_r, num_angles)

    # 6) 플롯용으로 (phi, r) 형태로 transpose
    # 오른쪽 예시처럼 y축=각도(phi), x축=r 이 되게 함
    sino_plot = sino.T  # shape: (num_angles, num_r)

    # 7) 축 단위(mm, deg) 만들기 위한 extent 계산
    h, w = img.shape
    # extent: 축 눈금이 의미하는 실제 좌표 범위 지정
    # 이미지 좌표(mm): 중심 기준 대략 [-W/2, W/2], [-H/2, H/2]
    x_extent_mm = (-(w / 2) * pixel_size_mm, (w / 2) * pixel_size_mm)
    y_extent_mm = (-(h / 2) * pixel_size_mm, (h / 2) * pixel_size_mm)

    # radon의 r축 샘플 개수 = sino.shape[0]
    num_r = sino.shape[0]
    # r축(mm): 대략 중심 기준으로 [-R, +R]로 놓고 보기 좋게 맞춤
    r_max_mm = (num_r / 2) * pixel_size_mm
    r_extent_mm = (-r_max_mm, r_max_mm)

    # 8) 시각화 (원본 + sinogram)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    ax1.set_title("Original")
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")
    ax1.imshow(
        img,
        cmap="gray",
        extent=(x_extent_mm[0], x_extent_mm[1], y_extent_mm[0], y_extent_mm[1]),
        interpolation="nearest", # 픽셀을 흐리게 보간하지 않고 픽셀 그대로를 보여줌
    )

    ax2.set_title("Radon transform (Sinogram)")
    ax2.set_xlabel("r [mm]")
    ax2.set_ylabel(r"$\phi$ [degree]")
    ax2.imshow(
        sino_plot,
        cmap="gray",
        extent=(r_extent_mm[0], r_extent_mm[1], theta[0], theta[-1]),
        origin="lower", # 각도 0이 아래로 옴
        aspect="auto", # 축 비율 자동, 안 쓰면 sinogram이 찌그러져 보일 수 있음
        interpolation="nearest", # 픽셀 그대로 보이게
    )

    plt.tight_layout() # 제목/축라벨이 겹치지 않게 자동 배치
    plt.show()

# ===== 사용 예시 =====
if __name__ == "__main__":
    show_sinogram_like_example(
        image_path=r"C:\Users\HEECHEOL\Desktop\grad\05_DL\02_DL\Q2\origin_circle.png",
        pixel_size_mm=1.0,      # <-- 실제 mm/px로 바꾸면 축이 딱 맞음
        n_angles=180,           # 오른쪽 예시 느낌
        #apply_circle_mask=True, # 원 밖 0 처리(권장)
    )