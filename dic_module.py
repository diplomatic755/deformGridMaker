# dic_mvp.py
# Базовый subset-DIC: ZNCC блок-сопоставление на сетке точек.
# Требует: opencv-python, numpy, pandas

import cv2
import numpy as np
import json, argparse
from pathlib import Path
import pandas as pd


def load_homography(video_path, px_per_mm, size_mm=22):
    calib_path = Path(video_path).with_suffix(".json")
    if not calib_path.exists():
        raise FileNotFoundError(f"Нет калибровки: {calib_path}")
    data = json.load(open(calib_path, "r", encoding="utf-8"))
    src = np.array(data["src_pts"], dtype=np.float32)
    W = int(round(size_mm * px_per_mm))
    H = W
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    Hmat = cv2.getPerspectiveTransform(src, dst)
    return Hmat, (W, H)


def warp(frame, Hmat, out_size):
    return cv2.warpPerspective(frame, Hmat, out_size, flags=cv2.INTER_CUBIC)


def make_grid(W, H, step_px, margin_px):
    xs = np.arange(margin_px, W - margin_px + 1, step_px, dtype=int)
    ys = np.arange(margin_px, H - margin_px + 1, step_px, dtype=int)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return grid_x, grid_y  # shape (Ny, Nx)


def zncc_match(ref, cur, cx, cy, w, search_r):
    """
    ref, cur: uint8 single-channel images (orthoplane)
    (cx,cy): center in ref (and expected in cur)
    w: half window (subset radius), patch size = (2w+1)^2
    search_r: search radius in cur around (cx,cy)
    Returns: (du, dv, score) subpixel displacement from (cx,cy) in pixels (u=x, v=y)
    """
    h, W = ref.shape
    # patch coordinates in ref
    x0 = cx - w;
    x1 = cx + w + 1
    y0 = cy - w;
    y1 = cy + w + 1
    if x0 < 0 or y0 < 0 or x1 > W or y1 > h:
        return np.nan, np.nan, -1.0
    templ = ref[y0:y1, x0:x1]
    # search ROI in cur
    sx0 = max(0, cx - search_r - w);
    sx1 = min(W, cx + search_r + w + 1)
    sy0 = max(0, cy - search_r - w);
    sy1 = min(h, cy + search_r + w + 1)
    roi = cur[sy0:sy1, sx0:sx1]
    if roi.shape[0] < templ.shape[0] or roi.shape[1] < templ.shape[1]:
        return np.nan, np.nan, -1.0

    # ZNCC ~ TM_CCOEFF_NORMED
    res = cv2.matchTemplate(roi, templ, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, maxloc = cv2.minMaxLoc(res)
    j, i = maxloc  # top-left in roi

    # subpixel refinement (quadratic fit around peak if possible)
    du_sp, dv_sp = 0.0, 0.0
    if 1 <= i < res.shape[0] - 2 and 1 <= j < res.shape[1] - 2:
        # parabolic fit in x and y separately on 3 samples
        def quad_peak(a, b, c):
            # peak of parabola through points (-1,a), (0,b), (1,c): x = 0.5*(a - c)/(a - 2b + c)
            denom = (a - 2 * b + c)
            if abs(denom) < 1e-9: return 0.0
            return 0.5 * (a - c) / denom

        # y-direction (rows)
        a, b, c = res[i - 1, j], res[i, j], res[i + 1, j]
        dv_sp = quad_peak(a, b, c)
        # x-direction (cols)
        a, b, c = res[i, j - 1], res[i, j], res[i, j + 1]
        du_sp = quad_peak(a, b, c)

    # convert to displacement in image coords
    # template position in roi: (i,j) is row,col. Convert to center shift wrt (cx,cy)
    # top-left of template in cur:
    cur_x0 = sx0 + j
    cur_y0 = sy0 + i
    # center:
    cur_cx = cur_x0 + w + du_sp
    cur_cy = cur_y0 + w + dv_sp

    du = cur_cx - cx
    dv = cur_cy - cy
    return du, dv, float(maxv)


def central_diff(arr, step_px):
    """Central differences for 2D array arr (Ny,Nx) → gradients wrt x and y in px units."""
    Ny, Nx = arr.shape
    dudx = np.zeros_like(arr, dtype=np.float32)
    dudy = np.zeros_like(arr, dtype=np.float32)
    # x-gradient
    dudx[:, 1:-1] = (arr[:, 2:] - arr[:, :-2]) / (2 * step_px)
    dudx[:, 0] = (arr[:, 1] - arr[:, 0]) / step_px
    dudx[:, -1] = (arr[:, -1] - arr[:, -2]) / step_px
    # y-gradient
    dudy[1:-1, :] = (arr[2:, :] - arr[:-2, :]) / (2 * step_px)
    dudy[0, :] = (arr[1, :] - arr[0, :]) / step_px
    dudy[-1, :] = (arr[-1, :] - arr[-2, :]) / step_px
    return dudx, dudy


def draw_overlay(gray, grid_x, grid_y, U, V, strain=None, mm_per_px=1.0, every=1):
    """Возвращает BGR картинку с визуализацией поля перемещений и εyy (если задан)."""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # heatmap εyy
    if strain is not None:
        eyy = strain.copy()
        eyy = np.clip(eyy, -0.2, 0.2)  # клиппинг для видимости
        eyy_norm = ((eyy - eyy.min()) / (eyy.max() - eyy.min() + 1e-9) * 255).astype(np.uint8)
        hm = cv2.applyColorMap(eyy_norm, cv2.COLORMAP_TURBO)
        hm = cv2.resize(hm, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
        vis = cv2.addWeighted(vis, 0.6, hm, 0.4, 0)

    # quiver (разрежаем для читаемости)
    for yidx in range(0, grid_y.shape[0], every):
        for xidx in range(0, grid_x.shape[1], every):
            x = int(grid_x[yidx, xidx]);
            y = int(grid_y[yidx, xidx])
            u = U[yidx, xidx];
            v = V[yidx, xidx]
            if np.isnan(u) or np.isnan(v):
                continue
            tip = (int(x + u), int(y + v))
            cv2.arrowedLine(vis, (x, y), tip, (0, 255, 255), 1, tipLength=0.25)
    # легенда масштаба (5 мм)
    scale_px = int(round(5 / mm_per_px))
    cv2.line(vis, (10, vis.shape[0] - 10), (10 + scale_px, vis.shape[0] - 10), (255, 255, 255), 2)
    cv2.putText(vis, "5 mm", (12 + scale_px, vis.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser(description="MVP DIC (subset-ZNCC) на сетке для 22x22 мм образца")
    ap.add_argument("-i", "--input", required=True, help="Путь к исходному видео")
    ap.add_argument("--scale", type=int, default=12, help="px на 1 мм (ортопроекция)")
    ap.add_argument("--subset", type=int, default=9, help="половина окна (w), итоговый патч = (2w+1)")
    ap.add_argument("--search", type=int, default=7, help="радиус поиска в пикселях")
    ap.add_argument("--grid_mm", type=float, default=1.0, help="шаг сетки в мм (центры подмножеств)")
    ap.add_argument("--margin_mm", type=float, default=1.0, help="отступ от краёв в мм")
    ap.add_argument("--stride", type=int, default=1, help="обрабатывать каждый N-й кадр")
    ap.add_argument("-o", "--output", default=None, help="имя выходного MP4")
    ap.add_argument("--csv", default=None, help="имя CSV со средними деформациями")
    ap.add_argument("--quiver_every", type=int, default=2, help="разрежение стрелок")
    args = ap.parse_args()

    Hmat, (W, H) = load_homography(args.input, args.scale, 22)
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Считываем первый кадр как опорный
    ok, f0 = cap.read()
    if not ok: raise RuntimeError("Пустое видео")
    ref = warp(f0, Hmat, (W, H))
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    # Грид центров (в пикселях)
    step_px = max(1, int(round(args.grid_mm * args.scale)))
    margin_px = max(args.subset + 2, int(round(args.margin_mm * args.scale)))
    grid_x, grid_y = make_grid(W, H, step_px, margin_px)
    Ny, Nx = grid_x.shape

    # Инициализация смещений
    U = np.zeros((Ny, Nx), dtype=np.float32)
    V = np.zeros((Ny, Nx), dtype=np.float32)
    valid = np.ones_like(U, dtype=bool)

    # Видео-писатель
    out_path = Path(args.output or Path(args.input).with_name(Path(args.input).stem + "_dic.mp4"))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / max(1, args.stride), (W, H))

    rows = []  # для CSV
    frame_idx = 0
    processed_idx = 0
    mm_per_px = 1.0 / args.scale

    # Основной цикл
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, frame = cap.read()
        if not ok: break
        if frame_idx % args.stride != 0:
            frame_idx += 1
            continue

        cur = warp(frame, Hmat, (W, H))
        cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

        # Для каждого центра — сопоставление
        U_new = np.full_like(U, np.nan, dtype=np.float32)
        V_new = np.full_like(V, np.nan, dtype=np.float32)
        score = np.full_like(U, -1.0, dtype=np.float32)

        for yi in range(Ny):
            for xi in range(Nx):
                # --- было ---
                # cx = int(grid_x[yi, xi] + U[yi, xi])
                # cy = int(grid_y[yi, xi] + V[yi, xi])

                # --- стало ---
                u_prev = U[yi, xi];
                v_prev = V[yi, xi]
                if not np.isfinite(u_prev): u_prev = 0.0
                if not np.isfinite(v_prev): v_prev = 0.0
                cx = int(round(grid_x[yi, xi] + u_prev))
                cy = int(round(grid_y[yi, xi] + v_prev))
                du, dv, sc = zncc_match(ref_gray, cur_gray, cx, cy, args.subset, args.search)

                #du, dv, sc = zncc_match(ref_gray, cur_gray, cx, cy, args.subset, args.search)
                if sc >= 0.5:  # порог совпадения
                    U_new[yi, xi] = du
                    V_new[yi, xi] = dv
                    score[yi, xi] = sc

        # простая медианная фильтрация выбросов
        U_f = cv2.medianBlur(np.nan_to_num(U_new, nan=0).astype(np.float32), 3)
        V_f = cv2.medianBlur(np.nan_to_num(V_new, nan=0).astype(np.float32), 3)
        U = U_f
        V = V_f
        U[np.isnan(U_new)] = np.nan
        V[np.isnan(V_new)] = np.nan

        # Градиенты и деформации (малые)
        dudx, dudy = central_diff(U, step_px)
        dvdx, dvdy = central_diff(V, step_px)
        exx = dudx
        eyy = dvdy
        gxy = 0.5 * (dvdx + dudy)

        # Средние значения по валидным точкам
        mask = ~np.isnan(U) & ~np.isnan(V)
        mean_exx = np.nanmean(exx[mask]) if np.any(mask) else np.nan
        mean_eyy = np.nanmean(eyy[mask]) if np.any(mask) else np.nan
        mean_gxy = np.nanmean(gxy[mask]) if np.any(mask) else np.nan
        time_s = processed_idx / (fps / max(1, args.stride))
        rows.append(
            {"frame": frame_idx, "t_s": time_s, "mean_exx": mean_exx, "mean_eyy": mean_eyy, "mean_gxy": mean_gxy})

        # Визуализация
        vis = draw_overlay(cur_gray, grid_x, grid_y, U, V, strain=eyy, mm_per_px=mm_per_px, every=args.quiver_every)
        writer.write(vis)

        processed_idx += 1
        frame_idx += 1

    cap.release()
    writer.release()

    csv_path = Path(args.csv or out_path.with_suffix(".csv"))
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[OK] Видео с полем деформаций: {out_path}")
    print(f"[OK] CSV со средними деформациями: {csv_path}")
    print("Пояснение: цвет — εyy (сжатие отриц.), стрелки — поле перемещений в пикселях.")


if __name__ == "__main__":
    main()
