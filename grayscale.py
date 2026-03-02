import numpy as np
import cv2
import fitz  # PyMuPDF
from PIL import Image, ImageFilter
from tqdm import tqdm

from utils import safe_print, is_valid_pdf, _recompress_streams_libdeflate


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
            _recompress_streams_libdeflate(output_path)
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
