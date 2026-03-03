import re
import shutil

import pikepdf
import fitz  # PyMuPDF
import numpy as np
import imagecodecs
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from config import (
    GRID_SIZE, COLOR_STD_THRESHOLD, BINARIZE_THRESHOLD,
    LIBDEFLATE_LEVEL, JP2K_THREADS, JP2K_WORKERS, MIN_IMAGE_SIZE,
)
from utils import safe_print, safe_remove, get_file_mb, is_valid_pdf, _recompress_streams_libdeflate, zlib_compress, tlog
from tiling_pass import detect_strict_color, detect_mono_or_hybrid

# 混合页面索引缓存: 首次扫描后记录哪些页面是混合页面(有文字/绘图)
# 页码索引是位置不变量，不受 pikepdf save 重编号 xref 的影响
_mixed_page_indices_cache = None  # None=未扫描, set()=已扫描


# ============================
# 图像大小估算 & 页级/流级回退守卫
# ============================

def _build_image_size_map(pdf_path):
    """构建 xref -> 图像流原始大小映射。"""
    size_map = {}
    try:
        with pikepdf.open(pdf_path) as pdf:
            for i, obj in enumerate(pdf.objects):
                if isinstance(obj, pikepdf.Stream) and obj.get("/Subtype") == "/Image":
                    try:
                        size_map[i] = len(obj.read_raw_bytes())
                    except:
                        pass
    except:
        pass
    return size_map


def _estimate_page_image_payloads(pdf_path):
    """按页估算图像载荷（每页按唯一 xref 计一次）。"""
    payloads = []
    try:
        size_map = _build_image_size_map(pdf_path)
        doc = fitz.open(pdf_path)
        for i in range(len(doc)):
            xrefs = sorted(set(im[0] for im in doc[i].get_images(full=True)))
            payloads.append(sum(size_map.get(x, 0) for x in xrefs))
        doc.close()
    except:
        return []
    return payloads


def rollback_worse_pages_by_image_payload(prev_pdf, cand_pdf, out_pdf, tolerance_bytes=0, max_rounds=3, skip_recompress=False):
    """将候选文件中"图像载荷变大"的页面回退为上一阶段页面（迭代收敛版）。

    说明：
    - 只以 prev_pdf 为唯一基准，不做相对原始输入页回退。
    - 因页面替换会改变资源引用关系，单轮回退后可能仍有残留放大页，
      这里迭代最多 max_rounds 轮，直到无放大页或达到上限。

    返回: (ok, total_rolled_back_pages)
    """
    try:
        prev_payloads = _estimate_page_image_payloads(prev_pdf)
        cand_payloads = _estimate_page_image_payloads(cand_pdf)
        if not prev_payloads or not cand_payloads:
            return False, 0
        if len(prev_payloads) != len(cand_payloads):
            return False, 0

        current_input = cand_pdf
        temp_outputs = []
        total_flagged = set()

        for round_idx in range(max(1, max_rounds)):
            cur_payloads = _estimate_page_image_payloads(current_input)
            if not cur_payloads or len(cur_payloads) != len(prev_payloads):
                break

            worse_pages = [
                idx for idx, (p0, p1) in enumerate(zip(prev_payloads, cur_payloads))
                if p1 > p0 + tolerance_bytes
            ]
            if not worse_pages:
                break

            for idx in worse_pages:
                total_flagged.add(idx)

            round_out = f"{out_pdf}.r{round_idx}"
            temp_outputs.append(round_out)

            cand_doc = fitz.open(current_input)
            prev_doc = fitz.open(prev_pdf)
            # 倒序替换，避免页码漂移
            for idx in sorted(worse_pages, reverse=True):
                cand_doc.delete_page(idx)
                cand_doc.insert_pdf(prev_doc, from_page=idx, to_page=idx, start_at=idx)
            cand_doc.save(round_out, garbage=4, deflate=True)
            cand_doc.close()
            prev_doc.close()
            if not skip_recompress:
                _recompress_streams_libdeflate(round_out)

            current_input = round_out

        final_payloads = _estimate_page_image_payloads(current_input)
        if final_payloads and len(final_payloads) == len(prev_payloads):
            remain_worse = [
                idx for idx, (p0, p1) in enumerate(zip(prev_payloads, final_payloads))
                if p1 > p0 + tolerance_bytes
            ]
            for idx in remain_worse:
                total_flagged.add(idx)

        if not total_flagged:
            for t in temp_outputs:
                safe_remove(t)
            return False, 0

        if current_input != out_pdf:
            shutil.copy2(current_input, out_pdf)

        for t in temp_outputs:
            if t != out_pdf:
                safe_remove(t)

        return is_valid_pdf(out_pdf), len(total_flagged)
    except:
        return False, 0




# ============================
# Phase 2: Safe Image Pass
# ============================

def _compress_single_xref(pdf, xref, target_quality, mixed_page_xrefs):
    """单个xref的压缩逻辑（接受已打开的pdf对象，避免重复打开PDF）"""
    image_obj = pdf.objects[xref]
    # Safety Checks - 硬 Mask 仍然跳过
    if "/Mask" in image_obj:
        return None
    # SMask: 检测全 255（完全不透明）的 trivial SMask，可安全忽略并继续处理
    if "/SMask" in image_obj:
        try:
            smask_obj = image_obj["/SMask"]
            smask_data = bytes(smask_obj.read_bytes())
            if smask_data != b'\xff' * len(smask_data):
                return None  # 非 trivial SMask，跳过
        except Exception:
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

            # 使用 libdeflate 压缩原始 1-bit 数据 (替代 zlib level 9)
            # PIL tobytes("raw") 对于 mode "1" 返回 packed bits (每行 byte 对齐)，符合 PDF 要求
            raw_bits = img_bin.tobytes()
            new_data = zlib_compress(raw_bits, LIBDEFLATE_LEVEL)

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


def _encode_single_image(xref, pil_image, orig_raw_size, target_quality, mixed_page_xrefs, is_dct_cmyk=False):
    """Phase B 编码函数：使用 imagecodecs 实现 GIL-free JPEG2000 编码，可安全多线程并发。
    与 _compress_single_xref 逻辑一致，但不依赖 pikepdf 对象（图片已在 Phase A 提取）。
    """
    # CMYK 图片：跳过灰度检测和二值化，直接走 JP2K 编码路径。
    # detect_strict_color 会将 CMYK 转 RGB 检测，低饱和度 CMYK 被误判为灰度后
    # 在混合页面触发二值化(1-bit DeviceGray)，完全破坏 CMYK 色彩信息。
    if pil_image.mode == "CMYK":
        try:
            img_arr = np.asarray(pil_image)
            # DCTDecode CMYK 使用反转通道约定(0=满墨, 255=无墨)，
            # 需反转为标准 PDF DeviceCMYK 约定(0=无墨, 255=满墨)，
            # 否则 JPXDecode 渲染器按标准约定解读会产生色彩反转。
            if is_dct_cmyk:
                img_arr = 255 - img_arr
            # CMYK 必须关闭 MCT（多分量变换仅适用于 RGB/YCbCr）
            if target_quality is not None:
                new_data = imagecodecs.jpeg2k_encode(
                    img_arr, level=target_quality, mct=False, numthreads=JP2K_THREADS
                )
            else:
                new_data = imagecodecs.jpeg2k_encode(
                    img_arr, level=0, mct=False, numthreads=JP2K_THREADS
                )
        except Exception:
            return None
        if len(new_data) >= orig_raw_size:
            return None
        return {
            "xref": xref,
            "data": new_data,
            "width": pil_image.width,
            "height": pil_image.height,
            "mode": "CMYK",
        }

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
            new_data = zlib_compress(raw_bits, LIBDEFLATE_LEVEL)
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

    # 1. 预扫描：识别哪些页面是 "混合页面" (有文字/矢量)
    # 这些页面上的灰度图必须在安全图片压缩阶段处理，不能留给破坏性切片阶段
    # 注意：缓存页码索引而非 xref，因为 pikepdf save 会重编号所有对象 xref
    tlog(f"I({desc}): 预扫描混合页面开始")
    global _mixed_page_indices_cache

    if _mixed_page_indices_cache is None:
        # 首次调用：fitz 全量扫描，确定哪些页面是混合页面
        mixed_page_indices = set()
        try:
            doc = fitz.open(input_path)
            for page_idx, page in enumerate(doc):
                has_text = len(page.get_text("text").strip()) > 0
                # get_drawings() 开销较大；有文字时已可判定为混合页，直接短路。
                has_drawings = False
                if not has_text:
                    has_drawings = len(page.get_drawings()) > 0
                if has_text or has_drawings:
                    mixed_page_indices.add(page_idx)
            doc.close()
        except:
            mixed_page_indices = set()
        _mixed_page_indices_cache = mixed_page_indices
        tlog(f"I({desc}): 页面索引缓存已建立, {len(mixed_page_indices)} 个混合页面")
    else:
        tlog(f"I({desc}): 使用页面索引缓存, {len(_mixed_page_indices_cache)} 个混合页面")

    # === Phase A: 单线程提取 (GIL-bound pikepdf) ===
    # 打开PDF，从缓存的混合页面索引中获取当前文件的图片 xref，然后提取所有图片
    tlog(f"I({desc}): PhaseA pikepdf.open 开始")
    extracted = []  # list of (xref, pil_image, orig_raw_size, is_dct_cmyk)
    trivial_smask_xrefs = set()  # 记录需要移除的全透明 SMask 图片 xref
    mixed_page_xrefs = set()
    try:
        with pikepdf.open(input_path) as pdf:
            # 从缓存的页面索引构建当前文件的 mixed_page_xrefs
            for page_idx in _mixed_page_indices_cache:
                page = pdf.pages[page_idx]
                resources = page.get("/Resources")
                if resources is None:
                    continue
                xobjects = resources.get("/XObject")
                if xobjects is None:
                    continue
                for name in xobjects:
                    obj = xobjects[name]
                    if not isinstance(obj, pikepdf.Stream):
                        continue
                    subtype = obj.get("/Subtype")
                    if subtype == "/Image":
                        mixed_page_xrefs.add(obj.objgen[0])
                    elif subtype == "/Form":
                        # Form XObject 可能嵌套图片
                        fr = obj.get("/Resources")
                        if fr is None:
                            continue
                        fx = fr.get("/XObject")
                        if fx is None:
                            continue
                        for fn in fx:
                            fo = fx[fn]
                            if isinstance(fo, pikepdf.Stream) and fo.get("/Subtype") == "/Image":
                                mixed_page_xrefs.add(fo.objgen[0])
            tlog(f"I({desc}): mixed_page_xrefs={len(mixed_page_xrefs)} (从页面索引构建)")

            tlog(f"I({desc}): PhaseA pikepdf.open 完成, 遍历对象")
            for i, obj in enumerate(pdf.objects):
                if isinstance(obj, pikepdf.Stream) and obj.get("/Subtype") == "/Image":
                    raw_size = len(obj.read_raw_bytes())
                    if raw_size < MIN_IMAGE_SIZE:
                        continue
                    # 硬 Mask（非 SMask）仍然跳过
                    if "/Mask" in obj:
                        continue
                    # SMask 检测：全 255 的 SMask（完全不透明）可安全移除
                    has_smask = "/SMask" in obj
                    if has_smask:
                        try:
                            smask_obj = obj["/SMask"]
                            smask_data = bytes(smask_obj.read_bytes())
                            # 检查是否全部为 255（完全不透明）
                            if smask_data == b'\xff' * len(smask_data):
                                trivial_smask_xrefs.add(i)
                                raw_size += len(smask_obj.read_raw_bytes())
                            else:
                                continue  # 非 trivial SMask，跳过
                        except Exception:
                            continue
                    try:
                        pdfimage = pikepdf.PdfImage(obj)
                        pil_image = pdfimage.as_pil_image()
                        if pil_image.width * pil_image.height > 100_000_000:
                            continue
                        # 标记 DCTDecode CMYK：JPEG CMYK 使用反转通道值约定(0=满墨)
                        # 而 FlateDecode/Indexed CMYK 使用标准约定(0=无墨)
                        is_dct_cmyk = (
                            pil_image.mode == "CMYK"
                            and str(obj.get("/Filter")) == "/DCTDecode"
                        )
                        extracted.append((i, pil_image, raw_size, is_dct_cmyk))
                    except Exception:
                        continue
    except Exception:
        return False
    tlog(f"I({desc}): PhaseA 提取完成, {len(extracted)} 张图片")
    if not extracted:
        return False

    # === Phase B: 多线程编码 (GIL-free numpy/PIL/zlib) ===
    tlog(f"I({desc}): PhaseB 多线程编码开始, {len(extracted)} 任务")
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
        for xref, pil_image, orig_raw_size, is_dct_cmyk in extracted:
            f = executor.submit(
                _encode_single_image, xref, pil_image, orig_raw_size,
                quality_db, mixed_page_xrefs, is_dct_cmyk
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
    tlog(f"I({desc}): PhaseB 编码完成, {len(results)} 结果")

    if results:
        try:
            tlog(f"I({desc}): PhaseC pikepdf.open 开始")
            with pikepdf.open(input_path, allow_overwriting_input=True) as pdf:
                for res in results:
                    obj = pdf.objects[res["xref"]]

                    # 移除已确认为 trivial 的 SMask 引用
                    if res["xref"] in trivial_smask_xrefs:
                        if "/SMask" in obj:
                            del obj["/SMask"]

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
                        if res["mode"] == "CMYK":
                            obj.ColorSpace = pikepdf.Name("/DeviceCMYK")
                        elif res["mode"] == "RGB":
                            obj.ColorSpace = pikepdf.Name("/DeviceRGB")
                        else:
                            obj.ColorSpace = pikepdf.Name("/DeviceGray")
                        obj.BitsPerComponent = 8
                        if "/Filter" in obj:
                            obj.Filter = pikepdf.Name("/JPXDecode")

                    # 清理通用属性
                    if "/DecodeParms" in obj:
                        del obj["/DecodeParms"]  # type: ignore
                    if "/Decode" in obj:
                        del obj["/Decode"]  # type: ignore
                    if "/ICCProfile" in obj:
                        del obj["/ICCProfile"]  # type: ignore
                tlog(f"I({desc}): PhaseC write-back 完成")
                tlog(f"I({desc}): PhaseC save 开始")
                pdf.save(output_path, compress_streams=True)
                tlog(f"I({desc}): PhaseC save 完成")
            return is_valid_pdf(output_path)
        except:
            return False
    return False
