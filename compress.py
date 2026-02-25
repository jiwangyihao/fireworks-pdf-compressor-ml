import sys
import os
import re
import io
import math
import shutil
import subprocess
import multiprocessing
import zlib  # Added for Flate compression
from pathlib import Path

# 添加当前目录到路径，确保能导入本地模块
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

import pikepdf
import fitz  # PyMuPDF
import numpy as np
import imagecodecs  # GIL-free JPEG2000 encoding (replaces PIL JPEG2000)
import cv2  # OpenCV for image enhancement
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# === ML Pipeline 可用性检查 (不预加载模型) ===
_ml_pipeline_available = None  # None=未检查, True=可用, False=不可用

def check_ml_pipeline_available():
    """检查ML管线是否可用（仅检查模型文件和依赖，不加载模型）"""
    global _ml_pipeline_available
    if _ml_pipeline_available is None:
        try:
            models_dir = SCRIPT_DIR / "models"
            required = [
                "slbr-watermark-detector-fp16.onnx",
                "real-esrgan-x4plus-128-fp16.onnx",
                "NAF-DPM_denoiser-fp16.onnx",
                "NAF-DPM_init_predictor-fp16.onnx",
            ]
            missing = [m for m in required if not (models_dir / m).exists()]
            if missing:
                safe_print(f"      [WARN] ML模型缺失: {', '.join(missing)}")
                _ml_pipeline_available = False
            else:
                import ml_pipeline  # noqa: F401 — verify importable
                _ml_pipeline_available = True
        except Exception as e:
            safe_print(f"      [WARN] ML管线不可用: {e}")
            _ml_pipeline_available = False
    return _ml_pipeline_available

# 全局配置
Image.MAX_IMAGE_PIXELS = None  # 禁用DecompressionBomb检查

# === 兼容性修复 ===
try:
    # type: ignore
    PDF_REDACT_IMAGE_REMOVE = fitz.PDF_REDACT_IMAGE_REMOVE
except AttributeError:
    PDF_REDACT_IMAGE_REMOVE = 2

# === 核心配置 (Integrated) ===
# 基础配置
SIZE_THRESHOLD_MB = 100
MIN_IMAGE_SIZE = 2048
CHUNK_SIZE = 100
CURVE_SIMPLIFY_THRESHOLD = 0.5
CPU_CORES = max(1, multiprocessing.cpu_count() - 1)

# JPEG2000 编码并行策略 (imagecodecs + OpenJPEG)
# numthreads: OpenJPEG 内部编码线程数 (释放 GIL)
# JP2K_WORKERS: ThreadPoolExecutor 并发编码任务数
# 最优组合: workers * numthreads ≈ CPU_CORES, 避免过度订阅
JP2K_THREADS = 4  # 每张图编码使用的 OpenJPEG 内部线程数
JP2K_WORKERS = max(2, CPU_CORES // JP2K_THREADS)  # 并发编码任务数

# 切片/灰度检测配置
GRID_SIZE = 20
GLOBAL_FORCE_MONO_THRESHOLD = 0.98
BLOCK_GRAY_PIXEL_THRESHOLD = 0.15
GRAY_LOWER_BOUND = 50
GRAY_UPPER_BOUND = 220
BINARIZE_THRESHOLD = 180
TILE_GRID_ROWS = 5
TILE_GRID_COLS = 5
TILE_CHECK_GRID = 4
COLOR_STD_THRESHOLD = 5.0  # 判定是否为彩色的阈值 (越小越容易被判定为彩色)

# 全局变量
lossy_report_list = []

# === 安全打印函数 (避免 Windows GBK 编码问题) ===
def safe_print(msg):
    """安全打印，处理 Windows 控制台的编码问题"""
    try:
        print(msg)
    except UnicodeEncodeError:
        # 移除无法编码的字符 (主要是 emoji)
        safe_msg = msg.encode('gbk', errors='replace').decode('gbk')
        print(safe_msg)


# ============================
# 辅助函数
# ============================
def safe_remove(path):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except:
            pass


def get_file_mb(path):
    if not os.path.exists(path):
        return 0
    return os.path.getsize(path) / (1024 * 1024)


def is_valid_pdf(path):
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < 1024:
        return False
    return True


# ============================
# 单色装饰模式检测 (逐页判定)
# ============================
def analyze_page_color_profile(page, gray_tolerance=10, gray_threshold=85, color_threshold=15):
    """
    分析单页的颜色特征
    返回: (is_mono_decorative, stats_dict)
    - is_mono_decorative: 该页是否为单色装饰页面（适合灰度化）
    """
    # 低分辨率采样
    pix = page.get_pixmap(matrix=fitz.Matrix(0.25, 0.25))
    img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    
    # NumPy向量化分析（替代逐像素Python循环，提速50-100x）
    arr = np.array(img)  # (H, W, 3) uint8
    total = arr.shape[0] * arr.shape[1]
    
    if total == 0:
        return False, {}
    
    r, g, b = arr[:,:,0].ravel(), arr[:,:,1].ravel(), arr[:,:,2].ravel()
    
    # 灰度判定: max(|R-G|, |G-B|, |R-B|) <= gray_tolerance
    diff_rg = np.abs(r.astype(np.int16) - g.astype(np.int16))
    diff_gb = np.abs(g.astype(np.int16) - b.astype(np.int16))
    diff_rb = np.abs(r.astype(np.int16) - b.astype(np.int16))
    max_diff = np.maximum(np.maximum(diff_rg, diff_gb), diff_rb)
    
    gray_count = int(np.count_nonzero(max_diff <= gray_tolerance))
    
    gray_ratio = gray_count / total * 100
    color_ratio = 100 - gray_ratio
    
    # 彩色像素色调统计
    color_mask = max_diff > gray_tolerance
    color_count = total - gray_count
    
    if color_count > 0:
        cr, cg, cb = r[color_mask], g[color_mask], b[color_mask]
        hue_counts = {
            'blue': int(np.count_nonzero((cb > cr) & (cb > cg))),
            'pink/red': int(np.count_nonzero((cr > cg) & (cr > cb))),
            'green': int(np.count_nonzero((cg > cr) & (cg > cb))),
        }
        hue_counts['other'] = color_count - sum(hue_counts.values())
        dominant_hue = max(hue_counts.items(), key=lambda x: x[1])
        hue_concentration = dominant_hue[1] / color_count * 100
    else:
        dominant_hue = ('none', 0)
        hue_concentration = 100
    
    stats = {
        'gray_ratio': gray_ratio,
        'color_ratio': color_ratio,
        'dominant_hue': dominant_hue[0],
        'hue_concentration': hue_concentration
    }
    
    # 单色装饰判定：
    # 1. 灰度占比 > 85%
    # 2. 彩色占比 < 15%
    # 3. 彩色部分的主色调集中度 > 60% (说明颜色单一)
    is_mono_decorative = (
        gray_ratio > gray_threshold and 
        color_ratio < color_threshold and 
        hue_concentration > 60
    )
    
    return is_mono_decorative, stats


def detect_mono_decorative_pages(pdf_path, sample_ratio=0.1, min_samples=15, max_samples=50):
    """
    检测PDF中的单色装饰页面
    
    返回: (has_mono_pages, pages_to_convert, stats)
    - has_mono_pages: 是否存在大量单色装饰页面 (>50%)
    - pages_to_convert: 需要灰度化的页面索引列表
    - stats: 统计信息
    """
    from collections import Counter
    import random
    
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        if total_pages == 0:
            doc.close()
            return False, [], {}
        
        # 确定采样数量
        sample_count = max(min_samples, min(max_samples, int(total_pages * sample_ratio)))
        sample_count = min(sample_count, total_pages)
        
        # 随机采样页面索引
        if sample_count >= total_pages:
            sample_indices = list(range(total_pages))
        else:
            sample_indices = sorted(random.sample(range(total_pages), sample_count))
        
        mono_pages_in_sample = []
        hue_counter = Counter()
        
        for page_num in sample_indices:
            page = doc[page_num]
            is_mono, stats = analyze_page_color_profile(page)
            
            if is_mono:
                mono_pages_in_sample.append(page_num)
            
            if stats.get('dominant_hue'):
                hue_counter[stats['dominant_hue']] += 1
        
        # 计算单色装饰页面比例
        mono_ratio = len(mono_pages_in_sample) / len(sample_indices) * 100
        
        # 如果采样中超过50%是单色装饰，则全量扫描
        pages_to_convert = []
        
        if mono_ratio > 50:
            # 全量扫描所有页面
            safe_print(f"      [SCAN] 检测到单色装饰模式 (采样中 {mono_ratio:.1f}% 符合)，正在全量扫描...")
            
            for page_num in tqdm(range(total_pages), desc="      Analyzing", leave=False):
                page = doc[page_num]
                is_mono, _ = analyze_page_color_profile(page)
                if is_mono:
                    pages_to_convert.append(page_num)
        
        doc.close()
        
        dominant_hue = hue_counter.most_common(1)[0][0] if hue_counter else 'none'
        
        summary_stats = {
            'total_pages': total_pages,
            'sample_count': len(sample_indices),
            'mono_in_sample': len(mono_pages_in_sample),
            'mono_ratio_sample': mono_ratio,
            'pages_to_convert': len(pages_to_convert),
            'dominant_hue': dominant_hue
        }
        
        has_significant_mono = len(pages_to_convert) > total_pages * 0.3  # 超过30%页面需要转换
        
        return has_significant_mono, pages_to_convert, summary_stats
        
    except Exception as e:
        return False, [], {'error': str(e)}


def enhance_document_image(img_array, mode='standard'):
    """
    文档图像增强 - 保守且安全的增强方式
    
    针对问题:
    1. 整体模糊 - 适度锐化
    2. 背景灰脏 - 温和的对比度调整
    3. 文字边缘模糊 - 自适应锐化
    
    处理流程:
    1. 轻度降噪 - 保留细节
    2. 对比度拉伸 - 扩展动态范围
    3. 自适应锐化 - 增强文字边缘
    4. 亮度微调 - 让背景稍微变白
    
    Args:
        img_array: numpy array (灰度图像, uint8)
        mode: 'standard' (标准), 'strong' (强力), 'mild' (温和)
    
    Returns:
        增强后的 numpy array
    """
    result = img_array.copy()
    
    # 根据模式调整参数
    if mode == 'strong':
        denoise_h = 5
        sharpen_amount = 0.4
        contrast_alpha = 1.15
        brightness_target = 245
    elif mode == 'mild':
        denoise_h = 3
        sharpen_amount = 0.2
        contrast_alpha = 1.05
        brightness_target = 240
    else:  # standard
        denoise_h = 4
        sharpen_amount = 0.3
        contrast_alpha = 1.1
        brightness_target = 242
    
    # ========================================
    # 1. 轻度降噪 (保留边缘)
    # ========================================
    # 使用较小的 h 值，只去除轻微噪点
    result = cv2.fastNlMeansDenoising(result, None, h=denoise_h, templateWindowSize=7, searchWindowSize=21)
    
    # ========================================
    # 2. 对比度拉伸 (Contrast Stretching)
    # ========================================
    # 将像素值拉伸到更宽的范围，但不要太激进
    min_val = np.percentile(result, 2)   # 2% 分位数
    max_val = np.percentile(result, 98)  # 98% 分位数
    
    if max_val > min_val + 20:  # 确保有足够的动态范围
        # 线性拉伸到 5-250 范围（留一点余量）
        result = np.clip((result - min_val) * 245.0 / (max_val - min_val) + 5, 0, 255).astype(np.uint8)
    
    # ========================================
    # 3. 温和的对比度增强
    # ========================================
    # 使用 CLAHE，但参数更保守
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
    result = clahe.apply(result)
    
    # ========================================
    # 4. 自适应锐化 (Unsharp Masking)
    # ========================================
    # 公式：sharpened = original + amount * (original - blurred)
    # 使用较小的 sigma 保留更多细节
    blurred = cv2.GaussianBlur(result, (0, 0), 1.5)
    result = cv2.addWeighted(result, 1 + sharpen_amount, blurred, -sharpen_amount, 0)
    
    # 第二次锐化 - 使用 PIL 的 UnsharpMask，更温和
    pil_img = Image.fromarray(result)
    pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=1, percent=80, threshold=3))
    result = np.array(pil_img)
    
    # ========================================
    # 5. 亮度微调 - 让背景稍微变白
    # ========================================
    # 计算当前背景亮度（取较亮区域的均值）
    bright_pixels = result[result > np.percentile(result, 70)]
    if len(bright_pixels) > 0:
        current_bg = np.mean(bright_pixels)
        
        # 如果背景不够白，适当提亮
        if current_bg < brightness_target:
            # 使用 gamma 校正提亮背景
            gamma = 0.95  # 轻微提亮
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
            result = cv2.LUT(result, table)
    
    # ========================================
    # 6. 最终对比度微调
    # ========================================
    result = cv2.convertScaleAbs(result, alpha=contrast_alpha, beta=0)
    
    # 确保输出在有效范围
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def convert_pages_to_grayscale(input_path, output_path, page_indices, dpi=150, enhance=True, use_ml=False):
    """
    将指定页面整页栅格化为灰度图片并替换原页面
    这会移除页面上的所有文字、矢量图形，只保留单张灰度图片
    目的是让这些页面能进入后续的二值化处理流程
    
    Args:
        input_path: 输入PDF路径
        output_path: 输出PDF路径
        page_indices: 需要灰度化的页面索引列表 (0-based)
        dpi: 栅格化分辨率 (默认150，平衡质量和文件大小)
        enhance: 是否应用图像增强
        use_ml: 是否使用ML增强 (True=ML增强, False=传统增强)
    """
    if not page_indices:
        return False
    
    try:
        # 使用 fitz (PyMuPDF) 进行页面栅格化
        doc = fitz.open(input_path)
        converted_count = 0
        total_pages = len(page_indices)
        
        page_set = set(page_indices)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        
        # === ML Pipeline Path ===
        # 使用5阶段流水线: Render -> SLBR(CPU) -> ESRGAN(GPU) -> NAF-DPM(GPU) -> DoxaPy(CPU)
        if enhance and use_ml:
            try:
                from ml_pipeline import PipelinedMLProcessor
                
                # 不传 models_dir，使用默认值 (SCRIPT_DIR / "models")
                # 这样在 EXE 中会自动找到 EXE 同级的 models 目录
                processor = PipelinedMLProcessor(
                    nafdpm_batch_size=32,
                    esrgan_overlap=8
                )
                
                # 自适应测试: 获取最优 batch_size (页数 > 20 时)
                if len(page_indices) > 20:
                    try:
                        from adaptive_config import get_optimal_config_for_pdf
                        batch_size, esrgan_overlap = get_optimal_config_for_pdf(
                            str(input_path), processor, force_benchmark=True)
                        processor.nafdpm_batch_size = batch_size
                        processor.esrgan_overlap = esrgan_overlap
                        print(f"[自适应] 最优 batch_size={batch_size}, overlap={esrgan_overlap}")
                    except Exception as e:
                        print(f"[自适应] 测试失败: {e}")
                
                results = processor.process_document(
                    str(input_path), page_indices, dpi=dpi)
                
                # 将灰度结果写入PDF
                for page_num in page_indices:
                    if page_num not in results:
                        continue
                    img_array = results[page_num]
                    page = doc[page_num]
                    h_img, w_img = img_array.shape[:2]
                    pgm_header = f"P5\n{w_img} {h_img}\n255\n".encode('ascii')
                    img_data = pgm_header + np.ascontiguousarray(img_array).tobytes()
                    page.clean_contents()
                    page_rect = page.rect
                    page.add_redact_annot(page_rect)
                    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_REMOVE)
                    page.insert_image(page_rect, stream=img_data)
                    converted_count += 1
                del results
                
            except Exception as e:
                import traceback
                safe_print(f"      [WARN] ML管线失败: {type(e).__name__}: {e}")
                safe_print(traceback.format_exc())
                safe_print("      [WARN] 回退到传统增强")
                use_ml = False  # 回退标记，进入下方传统路径
        
        # === 传统增强 / 无增强路径 (也用于ML回退) ===
        if not (enhance and use_ml):
            desc = "      传统增强" if enhance else "      栅格化"
            pbar = tqdm(page_indices, desc=desc,
                        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
            for page_num in pbar:
                if page_num >= len(doc):
                    continue
                page = doc[page_num]
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
                if enhance:
                    img_array = enhance_document_image(img_array)
                h_img, w_img = img_array.shape[:2]
                pgm_header = f"P5\n{w_img} {h_img}\n255\n".encode('ascii')
                img_data = pgm_header + np.ascontiguousarray(img_array).tobytes()
                page.clean_contents()
                page_rect = page.rect
                page.add_redact_annot(page_rect)
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_REMOVE)
                page.insert_image(page_rect, stream=img_data)
                converted_count += 1
            pbar.close()
        
        if converted_count > 0:
            doc.save(output_path, garbage=4, deflate=True)
            doc.close()
            return is_valid_pdf(output_path)
        else:
            doc.close()
            return False
        
    except Exception as e:
        try:
            print(f"      [WARN] 灰度转换失败: {e}")
        except UnicodeEncodeError:
            print(f"      [WARN] Grayscale conversion failed: {e}")
        return False


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


def extract_raw_jpx(data):
    """从 JP2 容器中提取原始 Codestream (如果存在)"""
    # JP2 Signature: 00 00 00 0C 6A 50 20 20 0D 0A 87 0A
    if data.startswith(b"\x00\x00\x00\x0cjP  \r\n\x87\n"):
        # It's a JP2 file, iterate boxes to find 'jp2c'
        pos = 0
        while pos < len(data) - 8:
            box_len = int.from_bytes(data[pos : pos + 4], "big")
            box_type = data[pos + 4 : pos + 8]

            if box_len == 0:
                # 0 means box extends to end of file
                if box_type == b"jp2c":
                    # 验证是否为 Codestream Start (SOC): FF 4F
                    if data[pos + 8 : pos + 10] == b"\xff\x4f":
                        return data[pos + 8 :]
                break

            if box_len == 1:
                # Large box (64-bit length), skip for simplicity or handle if needed
                # JP2 usually doesn't use this for small images
                pass

            if box_type == b"jp2c":
                if data[pos + 8 : pos + 10] == b"\xff\x4f":
                    return data[pos + 8 : pos + box_len]

            pos += box_len

    # 如果不是 JP2 容器，或者没找到 jp2c，或者已经是 Raw (FF 4F)
    if data.startswith(b"\xff\x4f"):
        return data

    # 如果无法提取，返回原始数据 (MuPDF 有时也能容忍完整的 JP2 数据)
    return data


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
# Phase 1/2: Core Functions
# ============================
def surgical_clean(input_path, output_path):
    print("      📋 正在执行清理...")
    try:
        with pikepdf.open(input_path) as pdf:
            if "/PieceInfo" in pdf.Root:
                del pdf.Root["/PieceInfo"]  # type: ignore
            if "/Metadata" in pdf.Root:
                del pdf.Root["/Metadata"]  # type: ignore
            for page in pdf.pages:
                if "/PieceInfo" in page:
                    del page["/PieceInfo"]  # type: ignore
                if "/Thumb" in page:
                    del page["/Thumb"]  # type: ignore
            pdf.remove_unreferenced_resources()
            pdf.save(
                output_path,
                compress_streams=True,
                object_stream_mode=pikepdf.ObjectStreamMode.generate,
            )
        return is_valid_pdf(output_path)
    except:
        return False


def get_gs_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_gs = os.path.join(base_dir, "gs", "bin", "gswin64c.exe")
    if os.path.exists(local_gs):
        return local_gs
    if shutil.which("gswin64c"):
        return "gswin64c"
    if shutil.which("gs"):
        return "gs"
    return None


def run_gs_level0(input_path, output_path, total_pages=0):
    gs_exe = get_gs_path()
    if not gs_exe:
        return False
    cmd = [
        gs_exe,
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dNOPAUSE",
        "-dBATCH",
        "-dSAFER",
        f"-sOutputFile={output_path}",
        "-dPDFSETTINGS=/prepress",
        "-dDownsampleColorImages=false",
        "-dDownsampleGrayImages=false",
        "-dDownsampleMonoImages=false",
        "-dAutoFilterColorImages=false",
        "-dAutoFilterGrayImages=false",
        "-dColorImageFilter=/FlateEncode",
        "-dGrayImageFilter=/FlateEncode",
        input_path,
    ]
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        pbar = None  # 延迟初始化进度条，以便显示原始输出

        if process.stdout:
            for line in process.stdout:
                # 检查是否开始处理页面 (GS输出通常包含 "Page X")
                if "Page" in line:
                    if pbar is None and total_pages > 0:
                        # 首次检测到页面处理，初始化进度条
                        pbar = tqdm(
                            total=total_pages,
                            desc="      🛡️ GS 重构",
                            unit="page",
                        )

                    if pbar:
                        try:
                            pbar.update(1)
                        except:
                            pass
                else:
                    # 在进度条出现前，显示 GS 的原始输出 (如 Loading font...)
                    if pbar is None:
                        # 使用 \r 覆盖当前行，避免刷屏
                        print(f"      [GS] {line.strip()}", end="\r")

        process.wait()
        if pbar:
            pbar.close()
        return process.returncode == 0 and is_valid_pdf(output_path)
    except:
        return False


# Regex Utils
NUM_RE = rb"[-+]?(?:\d*\.\d+|\d+)"
L_PATTERN = re.compile(rb"(" + NUM_RE + rb")(\s+)(" + NUM_RE + rb")(\s+)(l)(?=\s|$)")
C_PATTERN = re.compile(
    rb"("
    + NUM_RE
    + rb")(\s+)("
    + NUM_RE
    + rb")(\s+)"
    + rb"("
    + NUM_RE
    + rb")(\s+)("
    + NUM_RE
    + rb")(\s+)"
    + rb"("
    + NUM_RE
    + rb")(\s+)("
    + NUM_RE
    + rb")(\s+)(c)(?=\s|$)"
)
VY_PATTERN = re.compile(
    rb"("
    + NUM_RE
    + rb")(\s+)("
    + NUM_RE
    + rb")(\s+)"
    + rb"("
    + NUM_RE
    + rb")(\s+)("
    + NUM_RE
    + rb")(\s+)([vy])(?=\s|$)"
)
M_PATTERN = re.compile(rb"(" + NUM_RE + rb")(\s+)(" + NUM_RE + rb")(\s+)(m)(?=\s|$)")
W_PATTERN = re.compile(rb"(" + NUM_RE + rb")(\s+)(w)(?=\s|$)")


def format_number(bytes_val, sig_figs):
    try:
        # 安全检查：过短的数字不处理
        if len(bytes_val) < 3:
            return bytes_val
        val = float(bytes_val)
        new_str = (
            "{:.0f}".format(val)
            if val.is_integer()
            else "{:.{p}g}".format(val, p=sig_figs)
        )
        if new_str.startswith("0."):
            new_str = new_str[1:]
        elif new_str.startswith("-0."):
            new_str = "-" + new_str[2:]
        new_bytes = new_str.encode("ascii")
        if len(new_bytes) < len(bytes_val):
            return new_bytes
        return bytes_val
    except:
        return bytes_val


def replace_2args(m, sig_figs):
    """处理 l, m 等 2 参数命令"""
    return (
        format_number(m.group(1), sig_figs)
        + m.group(2)
        + format_number(m.group(3), sig_figs)
        + m.group(4)
        + m.group(5)
    )


def replace_w(m, sig_figs):
    """处理 w (线宽) 命令"""
    return format_number(m.group(1), sig_figs) + m.group(2) + m.group(3)


def replace_vy(m, sig_figs):
    """处理 v, y 曲线命令 (4 参数)"""
    n1 = format_number(m.group(1), sig_figs)
    n2 = format_number(m.group(3), sig_figs)
    n3 = format_number(m.group(5), sig_figs)
    n4 = format_number(m.group(7), sig_figs)
    op = m.group(9)
    return (
        n1
        + m.group(2)
        + n2
        + m.group(4)
        + n3
        + m.group(6)
        + n4
        + m.group(8)
        + op
    )


def replace_c_v4(m, sig_figs):
    """处理 c 曲线命令 (6 参数)"""
    nums = [format_number(m.group(i), sig_figs) for i in range(1, 12, 2)]
    spaces = [m.group(i) for i in range(2, 13, 2)]
    return b"".join(n + s for n, s in zip(nums, spaces)) + m.group(13)


def replace_c_smart(m, sig_figs):
    """智能曲线简化：如果曲线足够小，可以替换为直线"""
    try:
        x1, y1 = float(m.group(1)), float(m.group(3))
        x2, y2 = float(m.group(5)), float(m.group(7))
        x3, y3 = float(m.group(9)), float(m.group(11))
        span_x = max(x1, x2, x3) - min(x1, x2, x3)
        span_y = max(y1, y2, y3) - min(y1, y2, y3)
        if span_x < CURVE_SIMPLIFY_THRESHOLD and span_y < CURVE_SIMPLIFY_THRESHOLD:
            n3 = format_number(m.group(9), sig_figs)
            n4 = format_number(m.group(11), sig_figs)
            return n3 + b" " + n4 + b" l"
    except:
        pass
    return replace_c_v4(m, sig_figs)


def process_regex_chunk(file_path, xref_list, sig_figs, enable_smart_c):
    results = {}
    inline_img_re = re.compile(rb"(^|\s)BI(\s|$)")
    try:
        with pikepdf.open(file_path) as pdf:
            for xref in xref_list:
                try:
                    stream_obj = pdf.objects[xref]
                    if not isinstance(stream_obj, pikepdf.Stream):
                        continue
                    subtype = str(stream_obj.get("/Subtype") or "")
                    if "/Image" in subtype or "/Font" in subtype:
                        continue
                    raw_data = stream_obj.read_bytes()
                    if inline_img_re.search(raw_data):
                        continue

                    d = L_PATTERN.sub(lambda m: replace_2args(m, sig_figs), raw_data)
                    d = VY_PATTERN.sub(lambda m: replace_vy(m, sig_figs), d)
                    d = M_PATTERN.sub(lambda m: replace_2args(m, sig_figs), d)
                    d = W_PATTERN.sub(lambda m: replace_w(m, sig_figs), d)
                    if enable_smart_c:
                        d = C_PATTERN.sub(lambda m: replace_c_smart(m, sig_figs), d)
                    else:
                        d = C_PATTERN.sub(lambda m: replace_c_v4(m, sig_figs), d)

                    if len(d) < len(raw_data):
                        results[xref] = d
                except:
                    continue
        return results
    except:
        return {}


def run_regex_pass(input_path, output_path, sig_figs, enable_smart_c, desc):
    try:
        candidate_xrefs = []
        with pikepdf.open(input_path) as pdf:
            for i, obj in enumerate(pdf.objects):
                if isinstance(obj, pikepdf.Stream):
                    subtype = str(obj.get("/Subtype") or "")
                    if "/Image" not in subtype and "/Font" not in subtype:
                        candidate_xrefs.append(i)

        chunks = [
            candidate_xrefs[i : i + CHUNK_SIZE]
            for i in range(0, len(candidate_xrefs), CHUNK_SIZE)
        ]
        all_results = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    process_regex_chunk, input_path, chunk, sig_figs, enable_smart_c
                ): chunk
                for chunk in chunks
            }
            for f in tqdm(
                as_completed(futures),
                total=len(chunks),
                desc=f"      🔥 {desc}",
                unit="chunk",
            ):
                if res := f.result():
                    all_results.update(res)

        if not all_results:
            return False
        with pikepdf.open(input_path) as pdf:
            for xref, data in all_results.items():
                pdf.objects[xref].write(data)
            pdf.remove_unreferenced_resources()
            pdf.save(
                output_path,
                compress_streams=True,
                object_stream_mode=pikepdf.ObjectStreamMode.generate,
            )
        return is_valid_pdf(output_path)
    except:
        return False


# ============================
# Phase 2: Safe Image Pass (Modified)
# ============================
def _compress_single_xref(pdf, xref, target_quality, mixed_page_xrefs):
    """单个xref的压缩逻辑（接受已打开的pdf对象，避免重复打开PDF）"""
    image_obj = pdf.objects[xref]
    # Safety Checks
    if "/SMask" in image_obj or "/Mask" in image_obj:
        return None

    try:
        pdfimage = pikepdf.PdfImage(image_obj)
        pil_image = pdfimage.as_pil_image()

        # 安全检查：如果图片过大，可能导致 MemoryError，直接跳过
        # 100M 像素以上 (e.g. 10000x10000)
        if pil_image.width * pil_image.height > 100_000_000:
            return None

    except:
        # jbig2dec missing or other extraction error -> skip image
        return None

    # 1. 严格灰度检测
    is_color = detect_strict_color(
        pil_image, grid_size=GRID_SIZE, threshold=COLOR_STD_THRESHOLD
    )

    # 逻辑修正：
    # 安全图片压缩阶段应处理：
    # 1. 所有彩色图片 (is_color = True)
    # 2. 位于 "混合页面" (有文字/矢量) 上的灰度图片 (is_color = False and xref in mixed_page_xrefs)
    #
    # 安全图片压缩阶段应跳过：
    # 1. 位于 "纯图片页面" 上的灰度图片 (交给切片/二值化阶段)

    if not is_color:
        if xref not in mixed_page_xrefs:
            # 这是一个灰度图，且不在混合页面上 -> 跳过，留给切片/二值化阶段
            return None

        # 否则：虽然是灰度，但在混合页面上，进行安全压缩
        # === 新增逻辑：混合页面中的小图，如果满足二值化条件，直接二值化 ===
        mode, _ = detect_mono_or_hybrid(pil_image)
        if mode == "MONO" and target_quality is not None:
            # 二值化处理
            # type: ignore
            img_bin = pil_image.convert("L").point(
                lambda x: 0 if x < BINARIZE_THRESHOLD else 255, "1"
            )

            # 使用 Flate (Zlib) 压缩原始 1-bit 数据
            # PIL tobytes("raw") 对于 mode "1" 返回 packed bits (每行 byte 对齐)，符合 PDF 要求
            raw_bits = img_bin.tobytes()
            new_data = zlib.compress(raw_bits, level=9)

            if len(new_data) >= len(image_obj.read_raw_bytes()):
                return None

            return {
                "xref": xref,
                "data": new_data,
                "width": img_bin.width,
                "height": img_bin.height,
                "mode": "1",  # 标记为二值图
            }
        # 如果是 Hybrid 灰度图，继续下方的 JP2 流程

    # 正常处理 (Color 或 Mixed-Hybrid-Gray)
    if pil_image.mode not in ["RGB", "L"]:
        pil_image = pil_image.convert("RGB")

    try:
        img_arr = np.asarray(pil_image)
        if target_quality is not None:
            new_data = imagecodecs.jpeg2k_encode(img_arr, level=target_quality, numthreads=JP2K_THREADS)
        else:
            new_data = imagecodecs.jpeg2k_encode(img_arr, level=0, numthreads=JP2K_THREADS)
    except Exception:
        # Fallback for PIL/OpenJPEG encoding errors (broken data stream)
        # Try saving as PNG then return original (skip compression) or retry logic
        return None

    if len(new_data) >= len(image_obj.read_raw_bytes()):
        return None

    return {
        "xref": xref,
        "data": new_data,
        "width": pil_image.width,
        "height": pil_image.height,
        "mode": pil_image.mode,
    }


def compress_image_worker(xref, target_quality, pdf_path, mixed_page_xrefs):
    """单图兼容接口：每次调用打开PDF（保留向后兼容）"""
    with pikepdf.open(pdf_path) as pdf:
        return _compress_single_xref(pdf, xref, target_quality, mixed_page_xrefs)


def compress_image_chunk_worker(xref_chunk, target_quality, pdf_path, mixed_page_xrefs, progress_callback=None):
    """批量处理一组xref，只打开一次PDF（减少96%的PDF解析开销）"""
    results = []
    try:
        with pikepdf.open(pdf_path) as pdf:
            for xref in xref_chunk:
                res = None
                try:
                    res = _compress_single_xref(pdf, xref, target_quality, mixed_page_xrefs)
                    if res is not None:
                        results.append(res)
                except Exception:
                    pass
                if progress_callback:
                    progress_callback(1, res is not None)
    except Exception:
        pass
    return results


def _encode_single_image(xref, pil_image, orig_raw_size, target_quality, mixed_page_xrefs):
    """Phase B 编码函数：使用 imagecodecs 实现 GIL-free JPEG2000 编码，可安全多线程并发。
    与 _compress_single_xref 逻辑一致，但不依赖 pikepdf 对象（图片已在 Phase A 提取）。
    """
    # 1. 严格灰度检测
    is_color = detect_strict_color(
        pil_image, grid_size=GRID_SIZE, threshold=COLOR_STD_THRESHOLD
    )

    if not is_color:
        if xref not in mixed_page_xrefs:
            # 灰度图不在混合页面 -> 留给切片/二值化阶段
            return None

        # 混合页面中的灰度图，检查是否可二值化
        mode, _ = detect_mono_or_hybrid(pil_image)
        if mode == "MONO" and target_quality is not None:
            # type: ignore
            img_bin = pil_image.convert("L").point(
                lambda x: 0 if x < BINARIZE_THRESHOLD else 255, "1"
            )
            raw_bits = img_bin.tobytes()
            new_data = zlib.compress(raw_bits, level=9)
            if len(new_data) >= orig_raw_size:
                return None
            return {
                "xref": xref,
                "data": new_data,
                "width": img_bin.width,
                "height": img_bin.height,
                "mode": "1",
            }

    # 正常处理 (Color 或 Mixed-Hybrid-Gray)
    if pil_image.mode not in ["RGB", "L"]:
        pil_image = pil_image.convert("RGB")

    try:
        img_arr = np.asarray(pil_image)
        if target_quality is not None:
            new_data = imagecodecs.jpeg2k_encode(img_arr, level=target_quality, numthreads=JP2K_THREADS)
        else:
            new_data = imagecodecs.jpeg2k_encode(img_arr, level=0, numthreads=JP2K_THREADS)
    except Exception:
        return None

    if len(new_data) >= orig_raw_size:
        return None

    return {
        "xref": xref,
        "data": new_data,
        "width": pil_image.width,
        "height": pil_image.height,
        "mode": pil_image.mode,
    }


def run_image_pass_safe(input_path, output_path, quality_db, desc):
    """安全图片压缩 Pass"""

    # 1. 预扫描：识别哪些 XREF 属于 "混合页面" (有文字/矢量)
    # 这些页面上的灰度图必须在安全图片压缩阶段处理，不能留给破坏性切片阶段
    mixed_page_xrefs = set()
    try:
        doc = fitz.open(input_path)
        for page in doc:
            has_text = len(page.get_text("text").strip()) > 0
            # get_drawings() 开销较大；有文字时已可判定为混合页，直接短路。
            has_drawings = False
            if not has_text:
                has_drawings = len(page.get_drawings()) > 0

            if has_text or has_drawings:
                # 这是一个混合页面，记录其所有图片 XREF
                img_list = page.get_images(full=True)
                for img in img_list:
                    mixed_page_xrefs.add(img[0])  # xref is index 0
        doc.close()
    except:
        pass  # 如果扫描失败，mixed_page_xrefs 为空，灰度图将全部跳过给切片阶段 (风险较小)

    # === Phase A: 单线程提取 (GIL-bound pikepdf) ===
    # 只打开PDF一次，提取所有图片为PIL对象 + 原始大小
    extracted = []  # list of (xref, pil_image, orig_raw_size)
    try:
        with pikepdf.open(input_path) as pdf:
            for i, obj in enumerate(pdf.objects):
                if isinstance(obj, pikepdf.Stream) and obj.get("/Subtype") == "/Image":
                    raw_size = len(obj.read_raw_bytes())
                    if raw_size < MIN_IMAGE_SIZE:
                        continue
                    # 跳过带mask的图片
                    if "/SMask" in obj or "/Mask" in obj:
                        continue
                    try:
                        pdfimage = pikepdf.PdfImage(obj)
                        pil_image = pdfimage.as_pil_image()
                        if pil_image.width * pil_image.height > 100_000_000:
                            continue
                        extracted.append((i, pil_image, raw_size))
                    except Exception:
                        continue
    except Exception:
        return False
    if not extracted:
        return False

    # === Phase B: 多线程编码 (GIL-free numpy/PIL/zlib) ===
    results = []
    opt_count = [0]
    pbar = tqdm(
        total=len(extracted),
        desc=f"      🎨 {desc} (智能混合)",
        unit="img",
    )

    def on_encode_done(future):
        """编码完成回调 (GIL-free操作完成后)"""
        try:
            res = future.result()
        except Exception:
            res = None
        pbar.update(1)
        if res is not None:
            opt_count[0] += 1
            pbar.set_postfix(opt=opt_count[0])

    with ThreadPoolExecutor(max_workers=JP2K_WORKERS) as executor:
        futures = []
        for xref, pil_image, orig_raw_size in extracted:
            f = executor.submit(
                _encode_single_image, xref, pil_image, orig_raw_size,
                quality_db, mixed_page_xrefs
            )
            f.add_done_callback(on_encode_done)
            futures.append(f)
        # 等待所有编码完成
        for f in futures:
            try:
                res = f.result()
            except Exception:
                res = None
            if res is not None:
                results.append(res)
    pbar.close()

    if results:
        try:
            with pikepdf.open(input_path, allow_overwriting_input=True) as pdf:
                for res in results:
                    obj = pdf.objects[res["xref"]]

                    if res.get("mode") == "1":
                        # 二值化图片 (FlateDecode)
                        obj.write(res["data"], filter=pikepdf.Name("/FlateDecode"))
                        obj.Width = res["width"]
                        obj.Height = res["height"]
                        obj.ColorSpace = pikepdf.Name("/DeviceGray")
                        obj.BitsPerComponent = 1
                    else:
                        # 普通彩色/灰度图片 (JPXDecode)
                        obj.write(res["data"], filter=pikepdf.Name("/JPXDecode"))
                        obj.Width = res["width"]
                        obj.Height = res["height"]
                        obj.ColorSpace = (
                            pikepdf.Name("/DeviceRGB")
                            if res["mode"] == "RGB"
                            else pikepdf.Name("/DeviceGray")
                        )
                        obj.BitsPerComponent = 8
                        if "/Filter" in obj:
                            obj.Filter = pikepdf.Name("/JPXDecode")

                    # 清理通用属性
                    if "/DecodeParms" in obj:
                        del obj["/DecodeParms"]  # type: ignore
                    if "/ICCProfile" in obj:
                        del obj["/ICCProfile"]  # type: ignore
                pdf.remove_unreferenced_resources()
                pdf.save(output_path, compress_streams=True)
            return is_valid_pdf(output_path)
        except:
            return False
    return False


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
        # 如果包含文字或矢量图，直接跳过 (不进行原位压缩，因为安全阶段可能已经处理过，或者用户希望保留原样)
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
            # 此时已确定页面无干扰元素，可以安全地重构页面
            if mode == "MONO":
                # 防护1：仅处理覆盖率较高的 MONO 图层，避免小面积图层误触发整页 clean_contents
                if coverage_ratio < 0.5:
                    continue

                # 防护2：跳过“纯黑承载层”（常见于带 alpha/叠层的资源页），避免输出整页黑屏
                gray_arr = np.asarray(pil_img.convert("L"), dtype=np.uint8)
                if int(gray_arr.max()) == 0:
                    continue

                # 全单色 -> 直接二值化重构 (不区分大小，只要是纯图页面的 Mono 均处理)
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
                # 二）is_large 逻辑：
                # 在进入切分分支后，如果不是纯二值化情况 (即 Hybrid)，
                # 只有 "大图" 才进行切片处理；小图直接跳过。
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
                    # Hybrid 小图 -> 跳过 (User requested: skip instead of in-place re-compress)
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
    # 文件名格式: temp_chunk_{id}.pdf
    def get_chunk_id(fname):
        # 找到最后一个 _ 和 .pdf 之间的数字
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

    for cf in chunk_files:
        os.remove(cf)

    safe_print(
        f"      [STAT] 切片统计: 修改={total_stats['pages_modified']} | 切片: 二值={total_stats['mono_tiles']} 混合={total_stats['hybrid_tiles']}"
    )
    return is_valid_pdf(output_path)


# ============================
# 主流程
# ============================
def process_file(input_path, idx, total):
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
                if current_file != input_path:
                    safe_remove(current_file)
                current_file = tmp_gs
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
        safe_print(f"      ║  [3] 跳过增强")
        safe_print(f"      ║      - 仅栅格化为灰度，不做任何增强处理")
        safe_print(f"      ╚══════════════════════════════════════════════════════════════╝")
        
        # 用户交互
        try:
            user_input = input("      --> 请选择 [1/2/3]，直接回车默认选择 2 (传统增强): ").strip()
            
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
            # 非交互模式，默认使用传统增强
            safe_print("      [AUTO] 非交互模式，使用传统增强...")
            if convert_pages_to_grayscale(current_file, tmp_gray, pages_to_convert, enhance=True, use_ml=False):
                if current_file != input_path:
                    safe_remove(current_file)
                current_file = tmp_gray
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
        ("V", (2, False), "矢量 L3", ".tmp_v3"),
        ("I", (40,), "图片 40dB", ".tmp_i40"),
        ("T", (40,), "切片 40dB", ".tmp_t40"),
        ("V", (2, True), "矢量 L4", ".tmp_v4"),
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
        elif stype == "I":
            success = run_image_pass_safe(lossy_working_file, next_file, args[0], desc)
        elif stype == "T":
            # Tiling pass
            success = run_tiling_pass(lossy_working_file, next_file, args[0], desc)

        if success:
            new_mb = get_file_mb(next_file)
            if new_mb > 0.01 and new_mb < get_file_mb(lossy_working_file):
                safe_print(f"      [DOWN] 成功: {new_mb:.2f} MB")
                if lossy_working_file != input_path:
                    safe_remove(lossy_working_file)
                lossy_working_file = next_file
            else:
                safe_print("      [SKIP] 无收益")
                safe_remove(next_file)
        else:
            safe_print("      [SKIP] 已跳过")

    # Final Save
    final_mb = get_file_mb(lossy_working_file)
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
        except:
            pass
    else:
        safe_print("      [SKIP] 未优化")
        if lossy_working_file != input_path:
            safe_remove(lossy_working_file)

    # Cleanup
    for s in stages:
        safe_remove(input_path + s[3])
    safe_remove(tmp_clean)
    safe_remove(tmp_gs)
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

    pdf_files = [
        os.path.join(r, f)
        for r, d, fs in os.walk(".")
        for f in fs
        if f.lower().endswith(".pdf") and "_opted" not in f and ".tmp" not in f
    ]

    for idx, f in enumerate(pdf_files, 1):
        process_file(f, idx, len(pdf_files))

    if lossy_report_list:
        print("\n" + "=" * 50)
        safe_print("[WARN] 以下文件触发了有损压缩 (阶段 2/3)，请务必检查内容完整性：")
        for orig, opt in lossy_report_list:
            print(f" - {orig} -> {opt}")
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
