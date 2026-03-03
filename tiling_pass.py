"""切片处理 + 色彩检测逻辑"""
import io
import threading

import numpy as np
import fitz
import imagecodecs
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    GRID_SIZE, COLOR_STD_THRESHOLD, GRAY_LOWER_BOUND, GRAY_UPPER_BOUND,
    BLOCK_GRAY_PIXEL_THRESHOLD, GLOBAL_FORCE_MONO_THRESHOLD, BINARIZE_THRESHOLD,
    TILE_GRID_ROWS, TILE_GRID_COLS, TILE_CHECK_GRID,
    JP2K_THREADS, JP2K_WORKERS, PDF_REDACT_IMAGE_REMOVE,
)
from utils import safe_print, get_file_mb, is_valid_pdf, _recompress_streams_libdeflate, tlog

# 纯图片页面缓存: 首次扫描后记录哪些页是纯图片页（无文字无绘图），后续调用直接复用
_pure_image_pages_cache = None  # None=未扫描, set()=已扫描


# ============================
# 灰度/色彩 严格检测逻辑
# ============================
def is_block_gray(block):
    if block.size == 0:
        return False
    mid_mask = (block > GRAY_LOWER_BOUND) & (block < GRAY_UPPER_BOUND)
    return (np.count_nonzero(mid_mask) / block.size) > BLOCK_GRAY_PIXEL_THRESHOLD


def detect_strict_color(pil_img, grid_size=20, threshold=5.0):
    """
    严格色彩检测：将图片切分为 grid_size x grid_size 网格
    只要有一个网格的 RGB 通道标准差均值超过阈值，即判定为彩色
    """
    if pil_img.mode not in ["RGB", "CMYK"]:
        return False  # 已经是灰度或二值

    # 转换为RGB numpy数组
    arr = np.array(pil_img.convert("RGB"))
    h, w, _ = arr.shape

    # 1. 全局快速检测 (避免明显彩色图浪费时间切网格)
    # 计算图像中心的 100x100 区域
    cy, cx = h // 2, w // 2
    center_sample = arr[
        max(0, cy - 50) : min(h, cy + 50), max(0, cx - 50) : min(w, cx + 50)
    ]
    if center_sample.size > 0:
        if np.mean(np.std(center_sample, axis=2)) > (threshold * 2):
            return True

    # 2. 网格检测
    h_step = max(h // grid_size, 1)
    w_step = max(w // grid_size, 1)

    for y in range(0, h, h_step):
        for x in range(0, w, w_step):
            # 提取 Block
            block = arr[y : y + h_step, x : x + w_step]
            if block.size == 0:
                continue

            # 计算该 Block 的色彩饱和度 (RGB通道间的标准差)
            # 对于纯灰度，R=G=B，std=0
            # 允许少量噪点 (threshold)
            block_sat = np.mean(np.std(block, axis=2))

            if block_sat > threshold:
                return True  # 发现彩色区块，判定为彩色

    return False  # 未发现彩色区块，判定为灰度


def detect_mono_or_hybrid(pil_img):
    gray = pil_img.convert("L")
    arr = np.array(gray)
    h, w = arr.shape

    mid_mask_global = (arr > 30) & (arr < 225)
    if arr.size == 0:
        return "MONO", arr
    gray_ratio = np.count_nonzero(mid_mask_global) / arr.size

    if gray_ratio < (1.0 - GLOBAL_FORCE_MONO_THRESHOLD):
        return "MONO", arr

    h_step, w_step = max(h // GRID_SIZE, 1), max(w // GRID_SIZE, 1)
    has_gray = False
    for y in range(0, h, h_step):
        for x in range(0, w, w_step):
            if is_block_gray(arr[y : y + h_step, x : x + w_step]):
                has_gray = True
                break
        if has_gray:
            break
    return ("HYBRID" if has_gray else "MONO"), arr


def detect_tile_mode(pil_img, grid_size=4):
    gray = pil_img.convert("L")
    arr = np.array(gray)
    h, w = arr.shape

    mid_mask_global = (arr > 30) & (arr < 225)
    if arr.size == 0:
        return "MONO", arr
    gray_ratio = np.count_nonzero(mid_mask_global) / arr.size

    if gray_ratio < (1.0 - GLOBAL_FORCE_MONO_THRESHOLD):
        return "MONO", arr

    h_step, w_step = max(h // grid_size, 1), max(w // grid_size, 1)
    has_gray = False
    for y in range(0, h, h_step):
        for x in range(0, w, w_step):
            if is_block_gray(arr[y : y + h_step, x : x + w_step]):
                has_gray = True
                break
        if has_gray:
            break
    return ("HYBRID" if has_gray else "MONO"), arr


def is_tile_color(pil_img):
    """检测切片是否彩色 (复用 strict color 逻辑，但针对小图优化)"""
    return detect_strict_color(pil_img, grid_size=2, threshold=5.0)


# ============================
# Tile Grid Computation (pure, thread-safe)
# ============================
def _compute_tile_grid(pil_img, target_rect, orig_stream_len, quality_db):
    """纯计算: 切片网格分析 + 编码。返回 (tiles_list, stats) 或 (None, {})。"""
    stats = {"mono": 0, "hybrid": 0}
    w, h = pil_img.size
    h_step = h / TILE_GRID_ROWS
    w_step = w / TILE_GRID_COLS

    new_tiles = []
    total_new_size = 0

    # 1. Grid Analysis
    grid_map = []
    for r in range(TILE_GRID_ROWS):
        row_data = []
        for c in range(TILE_GRID_COLS):
            y0, x0 = int(r * h_step), int(c * w_step)
            y1 = int((r + 1) * h_step) if r < TILE_GRID_ROWS - 1 else h
            x1 = int((c + 1) * w_step) if c < TILE_GRID_COLS - 1 else w

            if x1 <= x0 or y1 <= y0:
                row_data.append(None)
                continue
            tile_img = pil_img.crop((x0, y0, x1, y1))
            tile_mode, _ = detect_tile_mode(tile_img)
            row_data.append(
                {"mode": tile_mode, "img": tile_img, "rect": (x0, y0, x1, y1)}
            )
        grid_map.append(row_data)

    # 2. Merge Logic
    visited = [
        [False for _ in range(TILE_GRID_COLS)] for _ in range(TILE_GRID_ROWS)
    ]

    for r in range(TILE_GRID_ROWS):
        for c in range(TILE_GRID_COLS):
            if visited[r][c] or not grid_map[r][c]:
                continue

            start_node = grid_map[r][c]
            current_mode = start_node["mode"]

            # Expand Right
            max_w = 0
            for k in range(c, TILE_GRID_COLS):
                node = grid_map[r][k]
                if not visited[r][k] and node and node["mode"] == current_mode:
                    max_w += 1
                else:
                    break

            # Expand Down
            max_h = 1
            for k_r in range(r + 1, TILE_GRID_ROWS):
                row_match = True
                for k_c in range(c, c + max_w):
                    node = grid_map[k_r][k_c]
                    if (
                        visited[k_r][k_c]
                        or not node
                        or node["mode"] != current_mode
                    ):
                        row_match = False
                        break
                if row_match:
                    max_h += 1
                else:
                    break

            # Mark Visited
            for i in range(max_h):
                for j in range(max_w):
                    visited[r + i][c + j] = True

            # Create Tile
            first_tile = grid_map[r][c]
            last_tile = grid_map[r + max_h - 1][c + max_w - 1]
            x0_big, y0_big = first_tile["rect"][0], first_tile["rect"][1]
            x1_big, y1_big = last_tile["rect"][2], last_tile["rect"][3]

            merged_img = pil_img.crop((x0_big, y0_big, x1_big, y1_big))

            stream_data = None
            if current_mode == "MONO":
                stats["mono"] += 1
                img_bin = merged_img.convert("L").point(
                    lambda x: 0 if x < BINARIZE_THRESHOLD else 255, "1"
                )
                buf = io.BytesIO()
                img_bin.save(buf, format="PNG", icc_profile=None)
                stream_data = buf.getvalue()
            else:
                stats["hybrid"] += 1
                if is_tile_color(merged_img):
                    _tile_arr = np.asarray(merged_img.convert("RGB"))
                else:
                    _tile_arr = np.asarray(merged_img.convert("L"))
                stream_data = imagecodecs.jpeg2k_encode(
                    _tile_arr, level=quality_db, codecformat='j2k',
                    numthreads=JP2K_THREADS,
                )

            total_new_size += len(stream_data)

            # Calculate PDF Coords
            scale_x = target_rect.width / w
            scale_y = target_rect.height / h
            off_x = x0_big * scale_x
            off_y = y0_big * scale_y
            tile_w_pdf = (x1_big - x0_big) * scale_x
            tile_h_pdf = (y1_big - y0_big) * scale_y

            tile_rect = fitz.Rect(
                target_rect.x0 + off_x,
                target_rect.y0 + off_y,
                target_rect.x0 + off_x + tile_w_pdf,
                target_rect.y0 + off_y + tile_h_pdf,
            )
            new_tiles.append((tile_rect, stream_data))

    # Check Size Inflation
    if total_new_size >= orig_stream_len:
        return None, {}

    return new_tiles, stats


# ============================
# Per-image computation (pure, thread-safe)
# ============================
def _compute_page_images(args):
    """Phase 2: 对一页的所有候选图做色彩检测 + 编码。返回修改列表。"""
    page_idx, images_data, target_quality = args
    modifications = []  # [(xref, redact_rect, [(tile_rect, stream_bytes)])]
    stats = {"mono": 0, "hybrid": 0}

    for xref, pil_img, target_rect, orig_len, coverage, all_rects in images_data:
        # 1. 严格灰度检测
        is_color = detect_strict_color(
            pil_img, grid_size=GRID_SIZE, threshold=COLOR_STD_THRESHOLD
        )
        if is_color:
            continue

        mode, _ = detect_mono_or_hybrid(pil_img)

        # 2. 灰度图处理策略
        if mode == "MONO":
            if coverage < 0.5:
                continue
            gray_arr = np.asarray(pil_img.convert("L"), dtype=np.uint8)
            if int(gray_arr.max()) == 0:
                continue

            img_bin = pil_img.convert("L").point(
                lambda x: 0 if x < BINARIZE_THRESHOLD else 255, "1"
            )
            buf = io.BytesIO()
            img_bin.save(buf, format="PNG", icc_profile=None)
            png_data = buf.getvalue()

            if len(png_data) >= orig_len:
                continue

            tiles = [(r, png_data) for r in all_rects]
            modifications.append((xref, target_rect, tiles))
            stats["mono"] += 1

        elif mode == "HYBRID":
            if coverage <= 0.5:
                continue
            tiles, tile_stats = _compute_tile_grid(
                pil_img, target_rect, orig_len, target_quality
            )
            if tiles:
                modifications.append((xref, target_rect, tiles))
                stats["mono"] += tile_stats.get("mono", 0)
                stats["hybrid"] += tile_stats.get("hybrid", 0)

    return page_idx, modifications, stats


# ============================
# Phase 3: Tiling Pass (3-phase: extract → compute → apply)
# ============================
def run_tiling_pass(input_path, output_path, target_quality, desc):
    """执行物理切片 Pass (针对灰度图)"""
    file_mb = get_file_mb(input_path)
    safe_print(f"      [TILE] {desc} (灰度切片/优化 {target_quality}dB)...")

    # ── Phase 1: 提取候选页面/图像数据 ──
    tlog(f"T({desc}): Phase1 fitz.open 开始")
    try:
        fitz.tools.set_stderr_file("/dev/null")
    except Exception:
        pass

    global _pure_image_pages_cache

    candidates = []  # [(page_idx, [(xref, pil_img, target_rect, orig_len, coverage, all_rects)])]
    doc = fitz.open(input_path)
    tlog(f"T({desc}): Phase1 fitz.open 完成, {len(doc)} 页")
    total_pages = len(doc)

    # 缓存命中: 直接跳过非纯图片页
    if _pure_image_pages_cache is not None:
        scan_pages = _pure_image_pages_cache
        tlog(f"T({desc}): Phase1 使用缓存, {len(scan_pages)} 纯图片页")
        if not scan_pages:
            doc.close()
            tlog(f"T({desc}): Phase1 缓存为空, 跳过")
            return False
    else:
        scan_pages = range(total_pages)

    pure_image_page_indices = set()

    for page_idx in scan_pages:
        page = doc[page_idx]

        img_list = page.get_images(full=True)
        if not img_list:
            continue

        # 纯图片页面才进入切片处理
        has_text = len(page.get_text("text").strip()) > 0
        has_drawings = len(page.get_drawings()) > 0
        if has_text or has_drawings:
            continue

        pure_image_page_indices.add(page_idx)

        page_images = []
        for img_info in list(img_list):
            xref = img_info[0]
            try:
                smask_xref = int(img_info[1]) if len(img_info) > 1 else 0
            except Exception:
                smask_xref = 0
            if smask_xref > 0:
                continue

            rects = page.get_image_rects(xref)
            if not rects:
                continue
            target_rect = rects[0]
            coverage = sum(r.get_area() for r in rects) / page.rect.get_area()

            pix = fitz.Pixmap(doc, xref)
            if pix.n >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            if pix.n == 1:
                pil_img = Image.frombytes('L', [pix.width, pix.height], pix.samples).convert('RGB')
            elif pix.n == 2:
                pix_no_alpha = fitz.Pixmap(fitz.csGRAY, pix)
                pil_img = Image.frombytes('L', [pix_no_alpha.width, pix_no_alpha.height], pix_no_alpha.samples).convert('RGB')
            else:
                pil_img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)

            orig_len = len(doc.xref_stream(xref))
            page_images.append((xref, pil_img, target_rect, orig_len, coverage, list(rects)))

        if page_images:
            candidates.append((page_idx, page_images))

    doc.close()

    # 更新缓存
    if _pure_image_pages_cache is None:
        _pure_image_pages_cache = pure_image_page_indices
        tlog(f"T({desc}): Phase1 缓存已建立, {len(pure_image_page_indices)} 纯图片页")

    tlog(f"T({desc}): Phase1 提取完成, {len(candidates)} 候选页")

    if not candidates:
        return False

    # ── Phase 2: 多线程并行计算 (色彩检测 + 编码) ──
    tlog(f"T({desc}): Phase2 多线程计算开始, {len(candidates)} 任务")
    all_modifications = {}  # page_idx → modifications
    total_stats = {"mono_tiles": 0, "hybrid_tiles": 0, "pages_modified": 0}
    pbar_lock = threading.Lock()
    done_pages = [0]

    pbar = tqdm(
        total=len(candidates),
        desc=f"      🧩 {desc} (切片)",
        unit="page",
    )

    tasks = [
        (page_idx, images_data, target_quality)
        for page_idx, images_data in candidates
    ]
    # 释放候选列表引用 (images_data 已被 tasks 持有)
    candidates = None

    workers = min(JP2K_WORKERS, len(tasks))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_compute_page_images, t): t[0] for t in tasks}
        for f in as_completed(futures):
            page_idx, modifications, stats = f.result()
            with pbar_lock:
                done_pages[0] += 1
                if modifications:
                    all_modifications[page_idx] = modifications
                    total_stats["pages_modified"] += 1
                    total_stats["mono_tiles"] += stats.get("mono", 0)
                    total_stats["hybrid_tiles"] += stats.get("hybrid", 0)
                pbar.update(1)
                pbar.set_postfix(mod=total_stats["pages_modified"])

    pbar.close()
    tasks = None
    tlog(f"T({desc}): Phase2 计算完成, {total_stats['pages_modified']} 页修改")

    if not all_modifications:
        return False

    # ── Phase 3: 单次写入 (顺序应用修改 + 保存) ──
    tlog(f"T({desc}): Phase3 fitz.open 开始")
    doc = fitz.open(input_path)
    for page_idx, modifications in sorted(all_modifications.items()):
        page = doc[page_idx]
        for xref, redact_rect, tiles in modifications:
            page.clean_contents()
            page.add_redact_annot(redact_rect)
            page.apply_redactions(images=PDF_REDACT_IMAGE_REMOVE)
            for t_rect, t_data in tiles:
                page.insert_image(t_rect, stream=t_data, overlay=True)

    tlog(f"T({desc}): Phase3 write-back 完成")
    tlog(f"T({desc}): Phase3 doc.save 开始")
    doc.save(output_path, garbage=4, deflate=True)
    tlog(f"T({desc}): Phase3 doc.save 完成")
    doc.close()

    # libdeflate 后处理 (silently handles /Pages loop)
    tlog(f"T({desc}): Phase3 _recompress 开始")
    _recompress_streams_libdeflate(output_path)
    tlog(f"T({desc}): Phase3 _recompress 完成")

    safe_print(
        f"      [STAT] 切片统计: 修改={total_stats['pages_modified']}"
        f" | 切片: 二值={total_stats['mono_tiles']} 混合={total_stats['hybrid_tiles']}"
    )
    return is_valid_pdf(output_path)
