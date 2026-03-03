import sys
import os
import shutil
import multiprocessing
from pathlib import Path

# 添加当前目录到路径，确保能导入本地模块
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

import pikepdf
import fitz  # PyMuPDF

from config import (
    check_ml_pipeline_available, SIZE_THRESHOLD_MB,
)
from utils import safe_print, safe_remove, get_file_mb, is_valid_pdf, _recompress_streams_libdeflate
from gs_pass import surgical_clean, run_gs_level0, rollback_worse_content_streams
from vector_pass import run_regex_pass, run_shape_pass
from tiling_pass import run_tiling_pass
from image_pass import (
    rollback_worse_pages_by_image_payload,
    run_image_pass_safe,
)
from grayscale import (
    detect_mono_decorative_pages, convert_pages_to_grayscale,
)


# 全局变量
lossy_report_list = []
large_file_report_list = []


def process_file(input_path, idx, total, unattended_mode=False):
    file_mb = get_file_mb(input_path)
    initially_under_threshold = file_mb < SIZE_THRESHOLD_MB
    base_name = os.path.basename(input_path)
    safe_print(f"\n[{idx}/{total}] Processing: {base_name} ({file_mb:.2f} MB)")

    current_file = input_path

    # Temp files
    tmp_clean = input_path + ".tmp_clean"
    tmp_gs = input_path + ".tmp_gs"
    tmp_img0 = input_path + ".tmp_img0"
    tmp_gray = input_path + ".tmp_gray"

    # --- Phase 1: Safe Phase ---
    safe_print("      [Phase 1] 安全模式 (清理 + GS + 无损)...")

    if initially_under_threshold:
        safe_print("      [OK] 源文件已较小 (<100MB)，仅运行安全模式。")

    if surgical_clean(input_path, tmp_clean):
        current_file = tmp_clean

    # 获取页数以显示 GS 进度条
    total_pages = 0
    try:
        with pikepdf.open(current_file) as pdf:
            total_pages = len(pdf.pages)
    except:
        pass

    # GS 可能会导致中文字体乱码或结构损坏，暂时禁用 GS 重构步骤
    # === 修复逻辑：尝试运行 GS，但如果结果损坏或文字乱码，自动回退 ===
    gs_success = run_gs_level0(current_file, tmp_gs, total_pages)
    if gs_success:
        _recompress_streams_libdeflate(tmp_gs)
        prev_mb_before_gs = get_file_mb(current_file)
        # 验证 GS 结果是否损坏 (MuPDF check)
        try:
            # 结构完整性 + 文字完整性检查 (合并打开两个PDF，减少I/O开销)
            check_doc = fitz.open(tmp_gs)
            orig_doc = fitz.open(current_file)
            gs_text = check_doc[0].get_text("text")
            orig_text = orig_doc[0].get_text("text")
            check_doc.close()
            orig_doc.close()

            keep_gs = True
            if len(orig_text.strip()) > 10:
                # 简单计算字符重合度
                set_orig = set(orig_text)
                set_gs = set(gs_text)
                common = set_orig.intersection(set_gs)
                # 如果重合字符数少于原字符种类的 50%，认为乱码
                if len(common) < len(set_orig) * 0.5:
                    safe_print("      [WARN] GS 文字损坏 (乱码检测)，回退 GS 步骤。")
                    keep_gs = False
            
            if keep_gs:
                # GS 回退：先流级回退，再页级回退，两轮结束后统一 libdeflate
                gs_candidate = tmp_gs

                # 1) 流级内容回退（先于页级）
                gs_content_guard_file = tmp_gs + ".tmp_guard_content"
                content_guard_ok, rolled_streams, affected_pages = rollback_worse_content_streams(
                    current_file, gs_candidate, gs_content_guard_file, tolerance_bytes=64,
                    skip_recompress=True,
                )
                if content_guard_ok and is_valid_pdf(gs_content_guard_file):
                    gs_candidate = gs_content_guard_file
                    safe_print(
                        f"      [GUARD] GS 流级内容回退: 回退 {rolled_streams} 条流 / {affected_pages} 页"
                    )

                # 2) 页级回退（后于流级）
                gs_guard_file = tmp_gs + ".tmp_guard_prev"
                gs_guard_ok, gs_worse_cnt = rollback_worse_pages_by_image_payload(
                    current_file, gs_candidate, gs_guard_file,
                    skip_recompress=True,
                )
                if gs_guard_ok and is_valid_pdf(gs_guard_file):
                    gs_candidate = gs_guard_file
                    safe_print(f"      [GUARD] GS 页级回退(迭代): 回退 {gs_worse_cnt} 页")

                # fitz.save(deflate=True) 仅压缩未压缩流，不会重编码已有 FlateDecode 流，
                # 因此页级回退后流质量 (libdeflate) 已保持，无需重压缩。

                # 严格相对上一阶段回退：守卫后仍不变小则不采用
                gs_mb = get_file_mb(gs_candidate)
                if gs_mb > 0.01 and gs_mb < prev_mb_before_gs:
                    if current_file != input_path:
                        safe_remove(current_file)
                    current_file = gs_candidate
                    safe_print(f"      [DOWN] GS 重构收益: {gs_mb:.2f} MB")
                    if gs_candidate != tmp_gs:
                        safe_remove(tmp_gs)
                    if gs_candidate != gs_guard_file:
                        safe_remove(gs_guard_file)
                    if gs_candidate != gs_content_guard_file:
                        safe_remove(gs_content_guard_file)
                else:
                    safe_print("      [SKIP] GS 无收益，回退上一阶段。")
                    safe_remove(tmp_gs)
                    safe_remove(gs_guard_file)
                    safe_remove(gs_content_guard_file)
            else:
                safe_remove(tmp_gs)

        except:
            safe_print("      [WARN] GS 结果似乎损坏 (MuPDF 检查失败)，回退 GS 步骤。")
            safe_remove(tmp_gs)
            # 保持 current_file 不变 (即跳过 GS)
    else:
         safe_remove(tmp_gs)

    if run_image_pass_safe(current_file, tmp_img0, None, "无损"):
        if current_file != input_path:
            safe_remove(current_file)
        current_file = tmp_img0

    safe_mb = get_file_mb(current_file)
    safe_print(f"      [DOWN] 安全模式结果: {safe_mb:.2f} MB")

    # 规则：如果源文件一开始就小于阈值，安全阶段结束后必须直接返回，不进入后续有损流程。
    if initially_under_threshold:
        safe_print("      [OK] 原始文件已低于阈值，安全阶段后直接结束。")
        if safe_mb < file_mb:
            try:
                shutil.move(current_file, input_path)
                safe_print(
                    f"      [SAVE] 覆盖原文件: {os.path.basename(input_path)} ({safe_mb:.2f} MB)"
                )
            except:
                pass
        else:
            safe_print("      [SKIP] 安全模式无收益。")
            if current_file != input_path:
                safe_remove(current_file)

        # Cleanup temps
        safe_remove(tmp_clean)
        safe_remove(tmp_gs)
        safe_remove(tmp_img0)
        safe_remove(tmp_gray)
        return

    if safe_mb < SIZE_THRESHOLD_MB:
        safe_print("      [OK] 安全优化已达目标大小，跳过有损阶段。")
        # Finalize and return early (using current_file as result)
        if safe_mb < file_mb:
            try:
                shutil.move(current_file, input_path)
                safe_print(
                    f"      [SAVE] 覆盖原文件: {os.path.basename(input_path)} ({safe_mb:.2f} MB)"
                )
            except:
                pass
        else:
            safe_print("      [SKIP] 安全模式无收益。")
            if current_file != input_path:
                safe_remove(current_file)

        # Cleanup temps
        safe_remove(tmp_clean)
        safe_remove(tmp_gs)
        safe_remove(tmp_img0)
        safe_remove(tmp_gray)
        return

    # --- Phase 2: Interleaved Optimization (User Requested) ---
    # 先检测单色装饰页面
    safe_print("      [SCAN] 分析颜色特征...")
    has_mono_pages, pages_to_convert, mono_stats = detect_mono_decorative_pages(current_file)
    
    if has_mono_pages and pages_to_convert:
        total_pages_mono = mono_stats.get('total_pages', 0)
        convert_count = len(pages_to_convert)
        dominant_hue = mono_stats.get('dominant_hue', 'unknown')
        
        # 估算时间 (根据GPU/CPU区分)
        ml_available = check_ml_pipeline_available()
        gpu_name = 'CPU'
        try:
            from ml_enhance import get_gpu_providers
            _, gpu_name = get_gpu_providers()
        except Exception:
            pass
        # GPU (DirectML/CUDA) ~0.5-1分钟/页, CPU ~3-5分钟/页
        time_per_page_ml = 1 if gpu_name != 'CPU' else 5
        time_per_page_traditional = 0.1  # 分钟
        estimated_ml_time = convert_count * time_per_page_ml
        estimated_traditional_time = convert_count * time_per_page_traditional
        
        safe_print(f"")
        safe_print(f"      ╔══════════════════════════════════════════════════════════════╗")
        safe_print(f"      ║  检测到 {convert_count}/{total_pages_mono} 页单色装饰页面 (主色调: {dominant_hue})")
        safe_print(f"      ╠══════════════════════════════════════════════════════════════╣")
        safe_print(f"      ║  这些页面含有装饰色（如蓝色边框），但主体为灰度内容。")
        safe_print(f"      ║  栅格化后可启用二值化压缩，显著减小文件体积。")
        safe_print(f"      ╠══════════════════════════════════════════════════════════════╣")
        safe_print(f"      ║  [提醒] 请先确认这些页面是否确实适合灰度化：")
        safe_print(f"      ║         - 若文件已是清晰灰度稿，通常不必再次灰度化")
        safe_print(f"      ║         - 若需要保留专色/彩色信息，不建议强制灰度化")
        safe_print(f"      ║         - 若文件含有矢量内容（如板写笔记、矢量绘图），不建议灰度化")
        safe_print(f"      ║           灰度化会将矢量画面栅格化为位图，破坏矢量信息且无法还原")
        safe_print(f"      ║  [建议] 若文档为双色套印、以白底黑字为主，且文件体积较大（如 >100MB）的书籍扫描件，")
        safe_print(f"      ║         通常推荐灰度化以提升压缩收益。")
        safe_print(f"      ║         如不确定，建议先选 [4] 不执行灰度化，并人工抽检后再决定。")
        safe_print(f"      ╠══════════════════════════════════════════════════════════════╣")
        safe_print(f"      ║  增强方式选择:")
        safe_print(f"      ║")
        safe_print(f"      ║  [1] ML增强 (ESRGAN + NAF-DPM + DoxaPy)")
        safe_print(f"      ║      - 效果最佳：文字锐利、背景纯净、消除扫描噪点")
        # 显示GPU加速状态
        if gpu_name != 'CPU':
            safe_print(f"      ║      - GPU加速: {gpu_name} (大幅提升速度)")
        else:
            safe_print(f"      ║      - 使用CPU推理 (较慢)")
        safe_print(f"      ║      - 预计耗时: ~{estimated_ml_time} 分钟 ({convert_count}页 × ~{time_per_page_ml}分钟/页)")
        safe_print(f"      ║")
        safe_print(f"      ║  [2] 传统增强 (降噪 + CLAHE + 锐化)")
        safe_print(f"      ║      - 效果一般：基础去噪和对比度调整")
        safe_print(f"      ║      - 预计耗时: ~{estimated_traditional_time:.1f} 分钟")
        safe_print(f"      ║")
        safe_print(f"      ║  [3] 执行灰度化但跳过增强")
        safe_print(f"      ║      - 仅栅格化为灰度，不做任何增强处理")
        safe_print(f"      ║")
        safe_print(f"      ║  [4] 不执行灰度化 (推荐用于不确定场景)")
        safe_print(f"      ║      - 保持原页面色彩结构，不进行栅格化")
        safe_print(f"      ╚══════════════════════════════════════════════════════════════╝")
        
        # 用户交互 / 无人值守默认策略
        if unattended_mode:
            safe_print("      [AUTO] 无人值守模式：默认不执行灰度化，请后续人工抽检文件实际状态。")
        else:
            try:
                user_input = input("      --> 请选择 [1/2/3/4]，直接回车默认选择 2 (传统增强): ").strip()

                if user_input == '1':
                    safe_print(f"      [ML] 使用ML增强管线处理 {convert_count} 页...")
                    if convert_pages_to_grayscale(current_file, tmp_gray, pages_to_convert, enhance=True, use_ml=True):
                        if current_file != input_path:
                            safe_remove(current_file)
                        current_file = tmp_gray
                        safe_print(f"      [OK] ML增强完成: {get_file_mb(current_file):.2f} MB")
                    else:
                        safe_print("      [WARN] ML增强过程出错。")
                elif user_input == '3':
                    safe_print(f"      [SKIP] 跳过增强，仅栅格化...")
                    if convert_pages_to_grayscale(current_file, tmp_gray, pages_to_convert, enhance=False):
                        if current_file != input_path:
                            safe_remove(current_file)
                        current_file = tmp_gray
                        safe_print(f"      [OK] 栅格化完成: {get_file_mb(current_file):.2f} MB")
                elif user_input == '4':
                    safe_print("      [SKIP] 按用户选择跳过灰度化，请人工核验页面真实色彩属性。")
                else:  # 默认选择 2
                    safe_print(f"      [传统] 使用传统增强处理 {convert_count} 页...")
                    if convert_pages_to_grayscale(current_file, tmp_gray, pages_to_convert, enhance=True, use_ml=False):
                        if current_file != input_path:
                            safe_remove(current_file)
                        current_file = tmp_gray
                        safe_print(f"      [OK] 传统增强完成: {get_file_mb(current_file):.2f} MB")
                    else:
                        safe_print("      [WARN] 传统增强过程出错。")
            except EOFError:
                # 非交互模式，默认不执行灰度化
                safe_print("      [AUTO] 非交互模式：默认不执行灰度化，请人工核验页面真实色彩属性。")
    else:
        safe_print("      [OK] 未检测到显著的单色装饰页面。")

    # L1 -> I 50 -> T 50 -> L2 -> I 45 -> T 45 ...
    safe_print(
        "      [Phase 2] 交错压缩 (矢量 + 图片 + 灰度切片)..."
    )

    stages = [
        ("V", (4, False), "矢量 L1", ".tmp_v1"),
        ("I", (50,), "图片 50dB", ".tmp_i50"),
        ("T", (50,), "切片 50dB", ".tmp_t50"),
        ("V", (3, False), "矢量 L2", ".tmp_v2"),
        ("I", (45,), "图片 45dB", ".tmp_i45"),
        ("T", (45,), "切片 45dB", ".tmp_t45"),
        ("V", (3, True), "矢量 L3a", ".tmp_v3a"),
        ("S", (), "形状 L3b", ".tmp_s3b"),
        ("I", (40,), "图片 40dB", ".tmp_i40"),
        ("T", (40,), "切片 40dB", ".tmp_t40"),
        ("V", (2, False), "矢量 L4", ".tmp_v4"),
        ("I", (35,), "图片 35dB", ".tmp_i35"),
        ("T", (35,), "切片 35dB", ".tmp_t35"),
    ]

    lossy_working_file = current_file

    for step_idx, (stype, args, desc, suffix) in enumerate(stages, 1):
        if get_file_mb(lossy_working_file) < SIZE_THRESHOLD_MB:
            break

        safe_print(f"      [STEP] 步骤 {step_idx}: {desc} ...")
        next_file = input_path + suffix
        success = False

        if stype == "V":
            success = run_regex_pass(
                lossy_working_file, next_file, args[0], args[1], desc
            )
        elif stype == "S":
            success = run_shape_pass(lossy_working_file, next_file, desc)
        elif stype == "I":
            success = run_image_pass_safe(lossy_working_file, next_file, args[0], desc)
        elif stype == "T":
            # Tiling pass
            success = run_tiling_pass(lossy_working_file, next_file, args[0], desc)

        if success:
            candidate_file = next_file

            # 页级回退守卫：相对“上一阶段”回退（严格迭代）
            if stype in ["I", "T"]:
                guard_file_prev = next_file + ".tmp_guard_prev"
                guard_ok_prev, worse_cnt_prev = rollback_worse_pages_by_image_payload(
                    lossy_working_file, next_file, guard_file_prev
                )
                if guard_ok_prev and is_valid_pdf(guard_file_prev):
                    candidate_file = guard_file_prev
                    safe_print(f"      [GUARD] 上一阶段回退(迭代): 回退 {worse_cnt_prev} 页")

            new_mb = get_file_mb(candidate_file)
            if new_mb > 0.01 and new_mb < get_file_mb(lossy_working_file):
                safe_print(f"      [DOWN] 成功: {new_mb:.2f} MB")
                if lossy_working_file != input_path:
                    safe_remove(lossy_working_file)
                lossy_working_file = candidate_file
                # next_file 不是最终采用文件时清理之
                if candidate_file != next_file:
                    safe_remove(next_file)
            else:
                safe_print("      [SKIP] 无收益")
                safe_remove(next_file)
                # guard 文件若存在也清理
                safe_remove(next_file + ".tmp_guard_prev")
        else:
            safe_print("      [SKIP] 已跳过")

    # Final Save
    final_mb = get_file_mb(lossy_working_file)
    final_report_name = base_name
    final_report_mb = file_mb
    if final_mb < file_mb:
        root, ext = os.path.splitext(input_path)
        optimized_path = f"{root}_opted{ext}"
        try:
            shutil.move(lossy_working_file, optimized_path)
            safe_print(
                f"      [DONE] 完成: {os.path.basename(optimized_path)} ({final_mb:.2f} MB)"
            )
            # 只有有损压缩才添加到报告列表
            lossy_report_list.append((base_name, os.path.basename(optimized_path)))
            final_report_name = os.path.basename(optimized_path)
            final_report_mb = final_mb
        except:
            pass
    else:
        safe_print("      [SKIP] 未优化")
        if lossy_working_file != input_path:
            safe_remove(lossy_working_file)
        final_report_name = base_name
        final_report_mb = get_file_mb(input_path)

    if final_report_mb > SIZE_THRESHOLD_MB:
        large_file_report_list.append((base_name, final_report_name, final_report_mb))

    # Cleanup
    for s in stages:
        safe_remove(input_path + s[3])
    safe_remove(tmp_clean)
    safe_remove(tmp_gs)
    safe_remove(tmp_gs + ".tmp_guard_prev")
    safe_remove(tmp_gs + ".tmp_guard_content")
    safe_remove(tmp_img0)
    safe_remove(tmp_gray)


def main():
    if sys.platform.startswith("win"):
        try:
            # type: ignore
            sys.stdout.reconfigure(encoding="utf-8")
        except:
            pass
    multiprocessing.freeze_support()
    safe_print("=== Fireworks Hybrid Engine (Interleaved Compression) ===")
    
    # GPU detection for ML acceleration
    try:
        from ml_enhance import get_gpu_providers
        providers, provider_name = get_gpu_providers()
        if provider_name == 'CPU':
            safe_print(f"  [GPU] 未检测到GPU加速，将使用CPU推理")
        else:
            safe_print(f"  [GPU] ML加速: {provider_name} ({providers[0]})")
    except Exception:
        safe_print("  [GPU] ML模块未加载，GPU检测跳过")

    # 支持命令行指定文件: python compress.py file1.pdf file2.pdf ...
    if len(sys.argv) > 1:
        pdf_files = [p for p in sys.argv[1:] if os.path.isfile(p) and p.lower().endswith(".pdf")]
    else:
        pdf_files = [
            os.path.join(r, f)
            for r, d, fs in os.walk(".")
            for f in fs
            if f.lower().endswith(".pdf") and "_opted" not in f and ".tmp" not in f
        ]

    global lossy_report_list, large_file_report_list
    lossy_report_list = []
    large_file_report_list = []

    unattended_mode = False
    if len(pdf_files) > 5:
        safe_print("\n[INFO] 检测到待处理文件超过 5 个，可开启无人值守模式。")
        safe_print("       无人值守模式下将不再询问用户意见，且检测到单色装饰模式时默认不进行灰度化。")
        try:
            choice = input("       --> 是否开启无人值守模式? [y/N]: ").strip().lower()
            unattended_mode = choice in ("y", "yes")
        except EOFError:
            unattended_mode = True
            safe_print("       [AUTO] 非交互环境：自动开启无人值守模式。")

    if unattended_mode:
        safe_print("[MODE] 当前运行模式：无人值守")
    else:
        safe_print("[MODE] 当前运行模式：交互")

    for idx, f in enumerate(pdf_files, 1):
        process_file(f, idx, len(pdf_files), unattended_mode=unattended_mode)

    if lossy_report_list:
        print("\n" + "=" * 50)
        safe_print("[WARN] 以下文件触发了有损压缩 (阶段 2/3)，请务必检查内容完整性：")
        for orig, opt in lossy_report_list:
            print(f" - {orig} -> {opt}")
        print("=" * 50 + "\n")

    if large_file_report_list:
        print("\n" + "=" * 50)
        safe_print(f"[WARN] 以下文件在完整压缩流程后仍大于 {SIZE_THRESHOLD_MB}MB，请重点复核：")
        for orig, final_name, final_mb in large_file_report_list:
            print(f" - {orig} -> {final_name} ({final_mb:.2f} MB)")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        safe_print(f"\n[ERROR] 运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input("\n按回车键退出...")
