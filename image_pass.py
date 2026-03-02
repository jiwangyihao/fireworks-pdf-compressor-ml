import re
import shutil

import pikepdf
import fitz  # PyMuPDF
import numpy as np
import deflate
import imagecodecs
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from config import (
    GRID_SIZE, COLOR_STD_THRESHOLD, BINARIZE_THRESHOLD,
    LIBDEFLATE_LEVEL, JP2K_THREADS, JP2K_WORKERS, MIN_IMAGE_SIZE,
)
from utils import safe_print, safe_remove, get_file_mb, is_valid_pdf, _recompress_streams_libdeflate
from tiling_pass import detect_strict_color, detect_mono_or_hybrid


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


def rollback_worse_pages_by_image_payload(prev_pdf, cand_pdf, out_pdf, tolerance_bytes=0, max_rounds=3):
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


def _get_page_content_stream_pairs(page_obj):
    """提取页面 /Contents 中的流对象序列。

    返回: [(slot, stream_obj), ...]
    - slot: "/Contents" (单流) 或数组下标 (多流)
    """
    pairs = []
    try:
        contents = page_obj.get("/Contents", None)
        if isinstance(contents, pikepdf.Stream):
            pairs.append(("/Contents", contents))
        elif isinstance(contents, pikepdf.Array):
            for idx, item in enumerate(contents):
                if isinstance(item, pikepdf.Stream):
                    pairs.append((idx, item))
    except:
        return []
    return pairs


_GS_REF_RE = re.compile(rb"/([^\s/<>\[\]\(\)%]+)\s+gs(?=\s|$)")
_XOBJ_REF_RE = re.compile(rb"/([^\s/<>\[\]\(\)%]+)\s+Do(?=\s|$)")
_CS_REF_RE = re.compile(rb"/([^\s/<>\[\]\(\)%]+)\s+(?:cs|CS)(?=\s|$)")

# 资源类型 → (regex, PDF资源子字典键名)
_RESOURCE_REF_PATTERNS = [
    (_GS_REF_RE, "/ExtGState"),
    (_XOBJ_REF_RE, "/XObject"),
    (_CS_REF_RE, "/ColorSpace"),
]


def _extract_gs_refs_from_stream(stream_obj):
    """提取内容流中使用的 gs 资源名集合（不含前导 /）。"""
    try:
        data = stream_obj.read_bytes()
        return set(m.decode("latin1", "ignore") for m in _GS_REF_RE.findall(data))
    except:
        return set()


def _extract_all_resource_refs(stream_obj):
    """提取内容流中所有资源引用，按类型分组。

    返回: {"/ExtGState": set, "/XObject": set, "/ColorSpace": set}
    """
    result = {}
    try:
        data = stream_obj.read_bytes()
        for regex, res_key in _RESOURCE_REF_PATTERNS:
            names = set(m.decode("latin1", "ignore") for m in regex.findall(data))
            if names:
                result[res_key] = names
    except:
        pass
    return result


def _get_page_extgstate_keys(page_obj):
    """提取页面 /Resources /ExtGState 的键名集合（不含前导 /）。"""
    keys = set()
    try:
        res = page_obj.get("/Resources", None)
        if isinstance(res, pikepdf.Dictionary):
            ext = res.get("/ExtGState", None)
            if isinstance(ext, pikepdf.Dictionary):
                for k in ext.keys():
                    keys.add(str(k).lstrip("/"))
    except:
        return set()
    return keys


def _get_page_resources_dict(page_obj):
    """获取页面资源字典，支持沿 /Parent 链继承查找。"""
    try:
        node = page_obj
        hops = 0
        while node is not None and hops < 32:
            res = node.get("/Resources", None)
            if isinstance(res, pikepdf.Dictionary):
                return res
            node = node.get("/Parent", None)
            hops += 1
    except:
        return None
    return None


def _get_page_extgstate_dict(page_obj):
    """获取页面 ExtGState 资源字典，不存在则返回 None。"""
    try:
        res = _get_page_resources_dict(page_obj)
        if isinstance(res, pikepdf.Dictionary):
            ext = res.get("/ExtGState", None)
            if isinstance(ext, pikepdf.Dictionary):
                return ext
    except:
        return None
    return None


def _ensure_page_own_extgstate_dict(page_obj):
    """确保页面拥有可写的 /Resources /ExtGState（避免改到继承字典）。"""
    try:
        inherited_res = _get_page_resources_dict(page_obj)

        # 1) 资源字典下沉到当前页
        if "/Resources" in page_obj and isinstance(page_obj.get("/Resources"), pikepdf.Dictionary):
            page_res = page_obj.get("/Resources")
        else:
            if isinstance(inherited_res, pikepdf.Dictionary):
                page_res = pikepdf.Dictionary(inherited_res)
            else:
                page_res = pikepdf.Dictionary()
            page_obj["/Resources"] = page_res

        # 2) ExtGState 下沉到当前页资源
        ext = page_res.get("/ExtGState", None)
        if isinstance(ext, pikepdf.Dictionary):
            own_ext = pikepdf.Dictionary(ext)
            page_res["/ExtGState"] = own_ext
            return own_ext

        own_ext = pikepdf.Dictionary()
        page_res["/ExtGState"] = own_ext
        return own_ext
    except:
        return None


def _inject_prev_extgstate_aliases(prev_page, cand_page, missing_refs, cand_pdf):
    """把 prev 页面缺失的 ExtGState 名称注入到 cand 页面资源，作为名称映射别名。"""
    if not missing_refs:
        return True
    try:
        prev_ext = _get_page_extgstate_dict(prev_page)
        cand_ext = _ensure_page_own_extgstate_dict(cand_page)
        if not isinstance(prev_ext, pikepdf.Dictionary) or not isinstance(cand_ext, pikepdf.Dictionary):
            return False

        for r in sorted(missing_refs):
            key = pikepdf.Name("/" + r)
            prev_obj = prev_ext.get(key, None)
            if prev_obj is None:
                prev_obj = prev_ext.get("/" + r, None)
            if prev_obj is None:
                return False
            cand_ext[key] = cand_pdf.copy_foreign(prev_obj)

        return True
    except:
        return False


def _inject_missing_resources(prev_page, cand_page, refs_by_type, cand_pdf):
    """通用资源注入：将 prev 页面中缺失的资源条目注入 cand 页面。

    refs_by_type: {"/XObject": {"Im0","Im1"}, "/ColorSpace": {"CS2"}, ...}
    仅注入 cand 页面资源字典中确实缺失的条目。
    返回 True 如果所有缺失条目均成功注入。
    """
    if not refs_by_type:
        return True
    try:
        prev_res = _get_page_resources_dict(prev_page)
        if not isinstance(prev_res, pikepdf.Dictionary):
            return False

        # 确保 cand 页面有自己的可写资源字典
        if "/Resources" in cand_page and isinstance(cand_page.get("/Resources"), pikepdf.Dictionary):
            cand_res = cand_page.get("/Resources")
        else:
            inherited = _get_page_resources_dict(cand_page)
            cand_res = pikepdf.Dictionary(inherited) if isinstance(inherited, pikepdf.Dictionary) else pikepdf.Dictionary()
            cand_page["/Resources"] = cand_res

        for res_key, names in refs_by_type.items():
            prev_sub = prev_res.get(res_key, None)
            if not isinstance(prev_sub, pikepdf.Dictionary):
                return False

            cand_sub = cand_res.get(res_key, None)
            if not isinstance(cand_sub, pikepdf.Dictionary):
                cand_sub = pikepdf.Dictionary()
                cand_res[res_key] = cand_sub

            for name in sorted(names):
                key = pikepdf.Name("/" + name)
                if key in cand_sub:
                    continue  # 已存在，无需注入
                prev_obj = prev_sub.get(key, None)
                if prev_obj is None:
                    return False
                cand_sub[key] = cand_pdf.copy_foreign(prev_obj)

        return True
    except:
        return False


def rollback_worse_content_streams(
    prev_pdf,
    cand_pdf,
    out_pdf,
    tolerance_bytes=64,
    safe_resource_check=True,
):
    """流级内容回退：仅回退相对上一阶段"明显变大"的内容流。

    设计要点：
    - 仅比较/回退每页 /Contents 中按顺序对应的流对象，不做页级替换。
    - 使用 tolerance_bytes 容忍编码抖动（例如 +1B 这类无意义差异）。

        返回: (ok, rolled_stream_count, affected_page_count)

        safe_resource_check:
        - True: 回退前校验 prev 流中引用的资源（ExtGState/XObject/ColorSpace）
            在 cand 页面的 /Resources 中全部可解析；缺失时从 prev 页注入；
            注入失败则跳过该流。
        - False: 不做该校验（不安全模式，仅用于对照实验）。
    """
    try:
        rolled_streams = 0
        affected_pages = set()

        with pikepdf.open(prev_pdf) as prev, pikepdf.open(cand_pdf) as cand:
            if len(prev.pages) != len(cand.pages):
                return False, 0, 0

            for page_idx in range(len(cand.pages)):
                prev_pairs = _get_page_content_stream_pairs(prev.pages[page_idx])
                cand_pairs = _get_page_content_stream_pairs(cand.pages[page_idx])
                if not prev_pairs or not cand_pairs:
                    continue

                # 仅按共同可映射的流数量比较，避免结构差异导致误替换。
                compare_n = min(len(prev_pairs), len(cand_pairs))
                for k in range(compare_n):
                    _, prev_stream = prev_pairs[k]
                    cand_slot, cand_stream = cand_pairs[k]
                    try:
                        prev_raw_len = len(prev_stream.read_raw_bytes())
                        cand_raw_len = len(cand_stream.read_raw_bytes())
                    except:
                        continue

                    if cand_raw_len > prev_raw_len + tolerance_bytes:
                        if safe_resource_check:
                            try:
                                cand_page = cand.pages[page_idx]
                                all_refs = _extract_all_resource_refs(prev_stream)
                                if all_refs:
                                    # 检查所有资源类型，收集缺失项
                                    cand_res = _get_page_resources_dict(cand_page)
                                    missing_by_type = {}
                                    for res_key, names in all_refs.items():
                                        cand_sub = cand_res.get(res_key, None) if isinstance(cand_res, pikepdf.Dictionary) else None
                                        existing = set()
                                        if isinstance(cand_sub, pikepdf.Dictionary):
                                            existing = set(str(k).lstrip("/") for k in cand_sub.keys())
                                        missing = names - existing
                                        if missing:
                                            missing_by_type[res_key] = missing

                                    if missing_by_type:
                                        ok = _inject_missing_resources(
                                            prev.pages[page_idx], cand_page, missing_by_type, cand
                                        )
                                        if not ok:
                                            continue

                                        # 注入后二次确认所有引用可解析
                                        cand_res2 = _get_page_resources_dict(cand_page)
                                        for res_key, names in all_refs.items():
                                            cand_sub2 = cand_res2.get(res_key, None) if isinstance(cand_res2, pikepdf.Dictionary) else None
                                            existing2 = set(str(k).lstrip("/") for k in cand_sub2.keys()) if isinstance(cand_sub2, pikepdf.Dictionary) else set()
                                            if not names.issubset(existing2):
                                                ok = False
                                                break
                                        if not ok:
                                            continue
                            except:
                                continue
                        try:
                            replaced_stream = cand.copy_foreign(prev_stream)
                            cand_page = cand.pages[page_idx]
                            if cand_slot == "/Contents":
                                cand_page["/Contents"] = replaced_stream
                            else:
                                arr = cand_page.get("/Contents", None)
                                if isinstance(arr, pikepdf.Array) and 0 <= cand_slot < len(arr):
                                    arr[cand_slot] = replaced_stream
                                else:
                                    continue
                            rolled_streams += 1
                            affected_pages.add(page_idx)
                        except:
                            continue

            if rolled_streams <= 0:
                return False, 0, 0

            cand.save(
                out_pdf,
                compress_streams=True,
                object_stream_mode=pikepdf.ObjectStreamMode.generate,
            )

        _recompress_streams_libdeflate(out_pdf)
        return is_valid_pdf(out_pdf), rolled_streams, len(affected_pages)
    except:
        return False, 0, 0


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
            new_data = bytes(deflate.zlib_compress(raw_bits, LIBDEFLATE_LEVEL))

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
            new_data = bytes(deflate.zlib_compress(raw_bits, LIBDEFLATE_LEVEL))
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
    extracted = []  # list of (xref, pil_image, orig_raw_size, is_dct_cmyk)
    trivial_smask_xrefs = set()  # 记录需要移除的全透明 SMask 图片 xref
    try:
        with pikepdf.open(input_path) as pdf:
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

    if results:
        try:
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
                pdf.remove_unreferenced_resources()
                pdf.save(output_path, compress_streams=True)
            _recompress_streams_libdeflate(output_path)
            return is_valid_pdf(output_path)
        except:
            return False
    return False
