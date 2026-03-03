"""逐阶段测量压缩率 + 提取对比区域截图，保留中间产物"""
import sys
import os
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

import fitz
from utils import safe_print, get_file_mb, is_valid_pdf, _recompress_streams_libdeflate, tlog
from gs_pass import surgical_clean, run_gs_level0, rollback_worse_content_streams
from vector_pass import run_regex_pass, run_shape_pass
from tiling_pass import run_tiling_pass
from image_pass import rollback_worse_pages_by_image_payload, run_image_pass_safe

# ── 配置 ──
INPUT_PDF = SCRIPT_DIR / "笔记1-23级.pdf"
OUT_DIR = SCRIPT_DIR / "measure_output"
COMPARE_PAGE = 5          # 第6页 (0-indexed)
CROP_Y_START = 0.10       # 高度 10%
CROP_Y_END   = 0.30       # 高度 30%
CROP_X_START = 0.10       # 宽度 10%
CROP_X_END   = 0.30       # 宽度 30%
RENDER_SCALE = 8          # 8x 分辨率渲染


def extract_region(pdf_path, label):
    """从 pdf_path 的第6页提取指定区域截图，保存到 OUT_DIR"""
    doc = fitz.open(str(pdf_path))
    page = doc[COMPARE_PAGE]
    rect = page.rect
    clip = fitz.Rect(
        rect.width * CROP_X_START,
        rect.height * CROP_Y_START,
        rect.width * CROP_X_END,
        rect.height * CROP_Y_END,
    )
    mat = fitz.Matrix(RENDER_SCALE, RENDER_SCALE)
    pix = page.get_pixmap(matrix=mat, clip=clip)
    out_path = OUT_DIR / f"compare_{label}.png"
    pix.save(str(out_path))
    doc.close()
    safe_print(f"  [IMG] {out_path.name} ({pix.width}x{pix.height})")


def snapshot(src_path, label):
    """复制中间产物到 OUT_DIR 并截图"""
    dst = OUT_DIR / f"{label}.pdf"
    shutil.copy2(str(src_path), str(dst))
    mb = get_file_mb(str(dst))
    safe_print(f"  [SNAP] {label}: {mb:.2f} MB")
    extract_region(str(dst), label)
    return mb


def run_step(stype, args, desc, current, suffix):
    """执行单步并返回新文件路径（失败返回 None）"""
    next_file = str(INPUT_PDF) + suffix
    success = False
    if stype == "V":
        success = run_regex_pass(current, next_file, args[0], args[1], desc)
    elif stype == "S":
        success = run_shape_pass(current, next_file, desc)
    elif stype == "I":
        success = run_image_pass_safe(current, next_file, args[0], desc)
    elif stype == "T":
        success = run_tiling_pass(current, next_file, args[0], desc)

    if not success:
        return None

    candidate = next_file
    # 图片/切片需要页级回退守卫
    if stype in ["I", "T"]:
        guard_file = next_file + ".tmp_guard_prev"
        ok, cnt = rollback_worse_pages_by_image_payload(current, next_file, guard_file)
        if ok and is_valid_pdf(guard_file):
            candidate = guard_file
            safe_print(f"      [GUARD] 回退 {cnt} 页")

    new_mb = get_file_mb(candidate)
    if new_mb > 0.01 and new_mb < get_file_mb(current):
        return candidate
    return None


def main():
    OUT_DIR.mkdir(exist_ok=True)
    input_path = str(INPUT_PDF)
    safe_print(f"=== 逐阶段压缩测量 ===")
    safe_print(f"输入: {INPUT_PDF.name} ({get_file_mb(input_path):.2f} MB)")

    results = {}

    # ── 0. 原始 ──
    results["0_original"] = snapshot(input_path, "0_original")

    # ── Phase 1: 无损阶段 ──
    safe_print("\n── Phase 1: 无损阶段 ──")
    tlog("MAIN: 无损阶段开始")
    current = input_path

    # Clean
    tmp_clean = input_path + ".m_clean"
    if surgical_clean(input_path, tmp_clean):
        current = tmp_clean

    # GS
    import pikepdf
    total_pages = 0
    try:
        with pikepdf.open(current) as pdf:
            total_pages = len(pdf.pages)
    except:
        pass

    tmp_gs = input_path + ".m_gs"
    gs_ok = run_gs_level0(current, tmp_gs, total_pages)
    if gs_ok:
        _recompress_streams_libdeflate(tmp_gs)
        prev_mb = get_file_mb(current)
        # 流级回退
        gs_cand = tmp_gs
        gs_cg = tmp_gs + ".m_cg"
        cg_ok, rs, ap = rollback_worse_content_streams(current, gs_cand, gs_cg, tolerance_bytes=64, skip_recompress=True)
        if cg_ok and is_valid_pdf(gs_cg):
            gs_cand = gs_cg
            safe_print(f"  [GUARD] GS 流级回退: {rs} 流 / {ap} 页")
        # 页级回退
        gs_gf = tmp_gs + ".m_gp"
        page_rollback_triggered = False
        gp_ok, gp_cnt = rollback_worse_pages_by_image_payload(current, gs_cand, gs_gf, skip_recompress=True)
        if gp_ok and is_valid_pdf(gs_gf):
            gs_cand = gs_gf
            page_rollback_triggered = True
            safe_print(f"  [GUARD] GS 页级回退: {gp_cnt} 页")
        # 仅在 fitz 页级回退触发时需要重压缩 (fitz save 可能改变流压缩质量)
        # 若仅 pikepdf 流级回退，所有流已为 libdeflate 质量，无需重压缩
        if page_rollback_triggered:
            _recompress_streams_libdeflate(gs_cand)
        gs_mb = get_file_mb(gs_cand)
        if gs_mb > 0.01 and gs_mb < prev_mb:
            current = gs_cand
            safe_print(f"  [GS] 收益: {gs_mb:.2f} MB")

    # Safe image
    tmp_img0 = input_path + ".m_img0"
    if run_image_pass_safe(current, tmp_img0, None, "无损"):
        current = tmp_img0

    results["1_lossless"] = snapshot(current, "1_lossless")
    tlog("MAIN: 无损阶段完成")

    # ── Phase 2: 有损交错 ──
    # 按 Level 分组
    levels = [
        ("L1", [
            ("V", (4, False), "矢量 L1", ".m_v1"),
            ("I", (50,), "图片 50dB", ".m_i50"),
            ("T", (50,), "切片 50dB", ".m_t50"),
        ]),
        ("L2", [
            ("V", (3, False), "矢量 L2", ".m_v2"),
            ("I", (45,), "图片 45dB", ".m_i45"),
            ("T", (45,), "切片 45dB", ".m_t45"),
        ]),
        ("L3a", [
            ("V", (3, True), "矢量 L3a", ".m_v3a"),
        ]),
        ("L3b", [
            ("S", (), "形状 L3b", ".m_s3b"),
            ("I", (40,), "图片 40dB", ".m_i40"),
            ("T", (40,), "切片 40dB", ".m_t40"),
        ]),
        ("L4", [
            ("V", (2, False), "矢量 L4", ".m_v4"),
            ("I", (35,), "图片 35dB", ".m_i35"),
            ("T", (35,), "切片 35dB", ".m_t35"),
        ]),
    ]

    level_idx = 2
    for level_name, steps in levels:
        safe_print(f"\n── {level_name} ──")
        tlog(f"MAIN: {level_name} 开始")

        for stype, args, desc, suffix in steps:
            safe_print(f"  [STEP] {desc}")
            tlog(f"MAIN: {level_name}/{desc} 开始")
            result = run_step(stype, args, desc, current, suffix)
            tlog(f"MAIN: {level_name}/{desc} 完成")
            if result:
                safe_print(f"  [DOWN] {get_file_mb(result):.2f} MB")
                current = result
            else:
                safe_print(f"  [SKIP] 无收益")

        snap_label = f"{level_idx}_{level_name}"
        results[snap_label] = snapshot(current, snap_label)
        level_idx += 1

    # ── 汇总 ──
    safe_print(f"\n{'='*50}")
    safe_print(f"{'阶段':<20} {'大小 (MB)':>10} {'压缩率':>10}")
    safe_print(f"{'-'*50}")
    orig_mb = results["0_original"]
    for label, mb in results.items():
        ratio = (1 - mb / orig_mb) * 100 if orig_mb > 0 else 0
        safe_print(f"{label:<20} {mb:>10.2f} {ratio:>9.1f}%")
    safe_print(f"{'='*50}")


if __name__ == "__main__":
    main()
