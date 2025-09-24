# mvp_grid_overlay.py
import cv2
import numpy as np
import json
import os
import argparse
from pathlib import Path

def ask_four_points(frame, win_name="Calibration (click 4 corners: TL, TR, BR, BL)"):
    pts = []
    clone = frame.copy()

    def on_mouse(event, x, y, flags, param):
        nonlocal pts, clone
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            cv2.circle(clone, (x, y), 6, (0, 255, 0), -1)
            cv2.imshow(win_name, clone)

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 900, int(900*frame.shape[0]/frame.shape[1]))
    cv2.imshow(win_name, clone)
    cv2.setMouseCallback(win_name, on_mouse)
    print("Кликните 4 угла образца по часовой: TL → TR → BR → BL. Нажмите Enter, когда готовы.....")
    while True:
        key = cv2.waitKey(10) & 0xFF
        if key in (13, 10) and len(pts) == 4:  # Enter
            break
        if key == 27:  # Esc
            pts = []
            break
    cv2.destroyWindow(win_name)
    return np.array(pts, dtype=np.float32)

def build_grid_overlay(width_px, height_px, step_px, thickness=1):
    """Возвращает прозрачную маску-сетку (RGBA) для наложения."""
    grid = np.zeros((height_px, width_px, 4), dtype=np.uint8)
    # вертикальные
    for x in range(0, width_px, step_px):
        cv2.line(grid, (x, 0), (x, height_px-1), (255, 255, 255, 140), thickness)
    # горизонтальные
    for y in range(0, height_px, step_px):
        cv2.line(grid, (0, y), (width_px-1, y), (255, 255, 255, 140), thickness)
    # рамка
    cv2.rectangle(grid, (0,0), (width_px-1, height_px-1), (255, 255, 255, 200), 2)
    return grid

def overlay_rgba(bg_bgr, overlay_rgba):
    """Накладывает RGBA поверх BGR."""
    b, g, r = cv2.split(bg_bgr)
    bg = cv2.merge((b, g, r, np.full(b.shape, 255, dtype=np.uint8)))  # BGRA
    alpha = overlay_rgba[..., 3:4] / 255.0
    out = (overlay_rgba.astype(np.float32) * alpha +
           bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    return out[..., :3]

def main(args):
    in_path = Path(args.input)
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть видео")

    # Первый кадр для калибровки
    ret, frame0 = cap.read()
    if not ret:
        raise RuntimeError("Пустое видео")

    # Попытка читать сохран калибровку
    calib_path = in_path.with_suffix(".json")
    H = None
    if calib_path.exists():
        try:
            data = json.load(open(calib_path, "r", encoding="utf-8"))
            src = np.array(data["src_pts"], dtype=np.float32)
            print("Загружена калибровка из", calib_path.name)
        except Exception:
            src = None
    else:
        src = None

    if src is None or src.shape != (4,2):
        src = ask_four_points(frame0)
        if src.size == 0:
            print("Калибровка отменена.")
            return
        json.dump({"src_pts": src.tolist()}, open(calib_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("Калибровка сохранена в", calib_path.name)

    # таргет плоскость (мм → пиксели)
    px_per_mm = args.scale
    W = int(round(22 * px_per_mm))
    Hh = int(round(22 * px_per_mm))
    dst = np.array([[0,0], [W-1,0], [W-1,Hh-1], [0,Hh-1]], dtype=np.float32)
    Hmat = cv2.getPerspectiveTransform(src, dst)

    # делаем сетку в ортоплоскости
    step_px = max(1, int(round(px_per_mm)))  # 1 мм
    grid_rgba = build_grid_overlay(W, Hh, step_px, thickness=args.thickness)

    # видео писатель
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # совместимый кодек
    out_path = Path(args.output if args.output else in_path.with_name(in_path.stem + "_grid.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, Hh), isColor=True)

    # Вернуть первый кадр к началу
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    gif_frames = []
    import math
    # чисто для отладки импорт, для получения гифки
    #import imageio.v2 as imageio if args.gif else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # коррекция перспективы в плоскость образца
        rect = cv2.warpPerspective(frame, Hmat, (W, Hh), flags=cv2.INTER_CUBIC)

        # Наложить сетку
        rect_with_grid = overlay_rgba(rect, grid_rgba)

        # В серый и обратно в BGR (для единообразия и записи)
        gray = cv2.cvtColor(rect_with_grid, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        writer.write(gray_bgr)

        # По желанию собираем GIF (с даунсемплингом по кадрам) если флаг на гиф будет тру
        if args.gif:
            if frame_idx % max(1, int(round(fps/args.gif_fps))) == 0:
                gif_frames.append(gray)  # 8-бит
        frame_idx += 1

        if args.preview and frame_idx % int(fps) == 0:
            cv2.imshow("preview", cv2.resize(gray_bgr, (min(800, W), int(min(800, W)*Hh/W))))
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"Готово: {out_path}")
    if args.gif and gif_frames:
        gif_path = out_path.with_suffix(".gif")
        imageio.mimsave(gif_path, gif_frames, duration=1/args.gif_fps, loop=0)
        print(f"GIF сохранён: {gif_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Наложение 1-мм сетки на видео с калибровкой по 4 углам")
    parser.add_argument("-i", "--input", required=True, help="Путь к видео (mp4/avi и т.п.)")
    parser.add_argument("-o", "--output", default=None, help="Путь к выходному mp4 (по умолчанию *_grid.mp4)")
    parser.add_argument("--scale", type=int, default=12, help="Пикселей на 1 мм в ортопроекции (дефолт 12 → видео 264×264)")
    parser.add_argument("--thickness", type=int, default=1, help="Толщина линий сетки в пикселях")
    parser.add_argument("--preview", action="store_true", help="Показывать превью раз в ~1 сек")
    parser.add_argument("--gif", action="store_true", help="Дополнительно собрать GIF (осторожно: большой размер)")
    parser.add_argument("--gif_fps", type=float, default=6.0, help="FPS для GIF (дефолт 6)")
    args = parser.parse_args()
    main(args)
