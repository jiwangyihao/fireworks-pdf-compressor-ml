"""切片处理 + 色彩检测逻辑"""
import os
import io

import numpy as np
import fitz
import imagecodecs
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from config import (
    GRID_SIZE, COLOR_STD_THRESHOLD, GRAY_LOWER_BOUND, GRAY_UPPER_BOUND,
    BLOCK_GRAY_PIXEL_THRESHOLD, GLOBAL_FORCE_MONO_THRESHOLD, BINARIZE_THRESHOLD,
    TILE_GRID_ROWS, TILE_GRID_COLS, TILE_CHECK_GRID,
    JP2K_THREADS, JP2K_WORKERS, PDF_REDACT_IMAGE_REMOVE,
)
from utils import safe_print, get_file_mb, is_valid_pdf, _recompress_streams_libdeflate


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
# Tiling Processor Class
# ============================
class TileProcessor:
    def __init__(self, page, xref, pil_img, page_num=0, quality_db=50):
        self.page = page
        self.xref = xref
        self.pil_img = pil_img
        self.page_num = page_num
        self.quality_db = quality_db  # 接收质量参数

    def process_and_replace(self):
        stats = {"mono": 0, "hybrid": 0}
        w, h = self.pil_img.size
        h_step = h / TILE_GRID_ROWS
        w_step = w / TILE_GRID_COLS

        rects = self.page.get_image_rects(self.xref)
        if not rects:
            return False, stats
        target_rect = rects[0]

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
                tile_img = self.pil_img.crop((x0, y0, x1, y1))
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

                merged_img = self.pil_img.crop((x0_big, y0_big, x1_big, y1_big))

                stream_data = None
                if current_mode == "MONO":
                    stats["mono"] += 1
                    # type: ignore
                    img_bin = merged_img.convert("L").point(
                        lambda x: 0 if x < BINARIZE_THRESHOLD else 255, "1"
                    )
                    buf = io.BytesIO()
                    img_bin.save(buf, format="PNG", icc_profile=None)
                    stream_data = buf.getvalue()
                else:
                    stats["hybrid"] += 1
                    # imagecodecs: GIL-free JPEG2000 编码，直接输出 J2K codestream
                    if is_tile_color(merged_img):
                        _tile_arr = np.asarray(merged_img.convert("RGB"))
                    else:
                        _tile_arr = np.asarray(merged_img.convert("L"))
                    stream_data = imagecodecs.jpeg2k_encode(
                        _tile_arr, level=self.quality_db, codecformat='j2k',
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
        orig_len = len(self.page.parent.xref_stream(self.xref))
        if total_new_size >= orig_len:
            return False, stats

        # Apply
        self.page.clean_contents()
        self.page.add_redact_annot(target_rect)
        self.page.apply_redactions(images=PDF_REDACT_IMAGE_REMOVE)
        for t_rect, t_data in new_tiles:
            self.page.insert_image(t_rect, stream=t_data, overlay=True)

        return True, stats


# ============================
# Phase 3: Tiling Pass
# ============================
def process_page_chunk_tiling(input_path, start_page, end_page, chunk_id, target_quality, progress_callback=None):
    # 使用稳健的临时文件名 (避免包含 .tmp 等干扰)
    base, _ = os.path.splitext(input_path)
    # 取 base 的 hash 或 简单清理，防止过长
    # 生成：current_dir/temp_chunk_{id}.pdf
    chunk_output = os.path.join(
        os.path.dirname(input_path), f"temp_chunk_{chunk_id}.pdf"
    )

    chunk_stats = {
        "mono_tiles": 0,
        "hybrid_tiles": 0,
        "pages_modified": 0,
        "pages_skipped": 0,
    }

    doc = fitz.open(input_path)
    new_doc = fitz.open()
    new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)
    doc.close()

    modified_count = 0
    for i in range(len(new_doc)):
        page = new_doc[i]
        img_list = page.get_images(full=True)
        if not img_list:
            chunk_stats["pages_skipped"] += 1
            if progress_callback:
                progress_callback(1, False)
            continue

        # 一）前置条件：只有 "纯图片页面" (无文字、无矢量) 才进入破坏性切片处理
        has_text = len(page.get_text("text").strip()) > 0
        has_drawings = len(page.get_drawings()) > 0
        if has_text or has_drawings:
            chunk_stats["pages_skipped"] += 1
            if progress_callback:
                progress_callback(1, False)
            continue

        page_modified = False
        for img_info in list(img_list):
            xref = img_info[0]

            # 关键修复：带 SMask/Mask 的图层常是透明叠层资源，
            # 对其做二值化可能导致整页被重写为近全黑图（第34页黑屏问题）。
            try:
                smask_xref = int(img_info[1]) if len(img_info) > 1 else 0
            except:
                smask_xref = 0
            if smask_xref > 0:
                continue

            rects = page.get_image_rects(xref)
            if not rects:
                continue

            # 计算图片占比，用于 Hybrid 模式下的决策
            is_large = (sum(r.get_area() for r in rects) / page.rect.get_area()) > 0.5

            pix = fitz.Pixmap(new_doc, xref)
            if pix.n >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            if pix.n == 1:
                pil_img = Image.frombytes('L', [pix.width, pix.height], pix.samples)
                pil_img = pil_img.convert('RGB')
            elif pix.n == 2:
                # 灰度+alpha → 去alpha转RGB
                pix_no_alpha = fitz.Pixmap(fitz.csGRAY, pix)
                pil_img = Image.frombytes('L', [pix_no_alpha.width, pix_no_alpha.height], pix_no_alpha.samples)
                pil_img = pil_img.convert('RGB')
            else:
                pil_img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)

            # 1. 严格灰度检测
            is_color = detect_strict_color(
                pil_img, grid_size=GRID_SIZE, threshold=COLOR_STD_THRESHOLD
            )
            if is_color:
                continue  # 彩色图跳过

            mode = detect_mono_or_hybrid(pil_img)[0]
            # 基于当前 xref 的覆盖率，避免小面积承载层触发整页重写
            coverage_ratio = (sum(r.get_area() for r in rects) / page.rect.get_area()) if rects else 0.0

            # 2. 灰度图处理策略
            if mode == "MONO":
                # 防护1：仅处理覆盖率较高的 MONO 图层
                if coverage_ratio < 0.5:
                    continue

                # 防护2：跳过"纯黑承载层"
                gray_arr = np.asarray(pil_img.convert("L"), dtype=np.uint8)
                if int(gray_arr.max()) == 0:
                    continue

                # 全单色 -> 直接二值化重构
                # type: ignore
                img_bin = pil_img.convert("L").point(
                    lambda x: 0 if x < BINARIZE_THRESHOLD else 255, "1"
                )
                buf = io.BytesIO()
                img_bin.save(buf, format="PNG", icc_profile=None)

                # 重构页面内容
                page.clean_contents()
                page.add_redact_annot(rects[0])
                page.apply_redactions(images=PDF_REDACT_IMAGE_REMOVE)
                for rect in rects:
                    page.insert_image(rect, stream=buf.getvalue(), overlay=True)
                page_modified = True
                chunk_stats["mono_tiles"] += 1

            elif mode == "HYBRID":
                if is_large:
                    processor = TileProcessor(
                        page,
                        xref,
                        pil_img,
                        page_num=(start_page + i),
                        quality_db=target_quality,
                    )
                    success, stats = processor.process_and_replace()
                    if success:
                        page_modified = True
                        chunk_stats["mono_tiles"] += stats["mono"]
                        chunk_stats["hybrid_tiles"] += stats["hybrid"]
                else:
                    pass

        if page_modified:
            modified_count += 1
            chunk_stats["pages_modified"] += 1
        else:
            chunk_stats["pages_skipped"] += 1
        if progress_callback:
            progress_callback(1, page_modified)

    # 降低垃圾回收级别，避免在 Chunk 阶段破坏引用
    new_doc.save(chunk_output, garbage=0, deflate=True)
    new_doc.close()
    _recompress_streams_libdeflate(chunk_output)
    return chunk_output, chunk_stats


def run_tiling_pass(input_path, output_path, target_quality, desc):
    """执行物理切片 Pass (针对灰度图)"""
    file_mb = get_file_mb(input_path)
    safe_print(f"      [TILE] {desc} (灰度切片/优化 {target_quality}dB)...")
    # 抑制 MuPDF 的 C 级输出
    try:
        fitz.tools.set_stderr_file(os.devnull)  # type: ignore
    except:
        pass

    doc = fitz.open(input_path)
    total_pages = len(doc)
    doc.close()

    chunk_size = (total_pages + JP2K_WORKERS - 1) // JP2K_WORKERS
    futures = []
    modified_count = [0]  # 用list包装以便在闭包中修改

    pbar = tqdm(
        total=total_pages,
        desc=f"      🧩 {desc} (切片)",
        unit="page",
    )

    def on_page_done(n, was_modified):
        """每处理完一页由worker线程回调（tqdm.update线程安全）"""
        pbar.update(n)
        if was_modified:
            modified_count[0] += 1
            pbar.set_postfix(mod=modified_count[0])

    with ThreadPoolExecutor(max_workers=JP2K_WORKERS) as executor:
        for i in range(JP2K_WORKERS):
            start = i * chunk_size
            end = min((i + 1) * chunk_size - 1, total_pages - 1)
            if start > end:
                break
            futures.append(
                executor.submit(
                    process_page_chunk_tiling, input_path, start, end, i, target_quality,
                    progress_callback=on_page_done
                )
            )

    chunk_files = []
    total_stats = {
        "mono_tiles": 0,
        "hybrid_tiles": 0,
        "pages_modified": 0,
        "pages_skipped": 0,
    }

    for f in futures:
        res, stats = f.result()
        if res:
            chunk_files.append(res)
            for k in total_stats:
                total_stats[k] += stats.get(k, 0)
    pbar.close()

    # 合并 (Robust Sort)
    def get_chunk_id(fname):
        return int(fname.split("_")[-1].replace(".pdf", ""))

    chunk_files.sort(key=get_chunk_id)

    if not chunk_files:
        return False

    final_doc = fitz.open()
    for cf in chunk_files:
        c_doc = fitz.open(cf)
        final_doc.insert_pdf(c_doc)
        c_doc.close()

    final_doc.save(output_path, garbage=4, deflate=True)
    final_doc.close()
    _recompress_streams_libdeflate(output_path)

    for cf in chunk_files:
        os.remove(cf)

    safe_print(
        f"      [STAT] 切片统计: 修改={total_stats['pages_modified']} | 切片: 二值={total_stats['mono_tiles']} 混合={total_stats['hybrid_tiles']}"
    )
    return is_valid_pdf(output_path)
