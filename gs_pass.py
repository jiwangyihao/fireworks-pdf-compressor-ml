"""GhostScript 重构 + surgical_clean"""
import os
import re
import shutil
import subprocess

import pikepdf
from tqdm import tqdm

from utils import safe_print, is_valid_pdf, _recompress_streams_libdeflate, libdeflate_compress_pdf, tlog


def surgical_clean(input_path, output_path):
    print("      📋 正在执行清理...")
    try:
        tlog("GS: surgical_clean pikepdf.open 开始")
        with pikepdf.open(input_path) as pdf:
            tlog("GS: surgical_clean pikepdf.open 完成")
            if "/PieceInfo" in pdf.Root:
                del pdf.Root["/PieceInfo"]  # type: ignore
            if "/Metadata" in pdf.Root:
                del pdf.Root["/Metadata"]  # type: ignore
            for page in pdf.pages:
                if "/PieceInfo" in page:
                    del page["/PieceInfo"]  # type: ignore
                if "/Thumb" in page:
                    del page["/Thumb"]  # type: ignore
            tlog("GS: surgical_clean libdeflate 开始")
            libdeflate_compress_pdf(pdf)
            tlog("GS: surgical_clean save 开始")
            pdf.save(
                output_path,
                compress_streams=True,
                object_stream_mode=pikepdf.ObjectStreamMode.generate,
            )
            tlog("GS: surgical_clean save 完成")
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
    tlog("GS: run_gs_level0 subprocess 开始")
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
        tlog(f"GS: run_gs_level0 subprocess 完成, returncode={process.returncode}")
        if pbar:
            pbar.close()
        return process.returncode == 0 and is_valid_pdf(output_path)
    except:
        return False


# ============================
# GS 流级内容回退
# ============================

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
    skip_recompress=False,
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

        tlog("GS: rollback pikepdf.open prev+cand 开始")
        with pikepdf.open(prev_pdf) as prev, pikepdf.open(cand_pdf) as cand:
            tlog(f"GS: rollback pikepdf.open 完成, prev={len(prev.pages)}p cand={len(cand.pages)}p")
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

            tlog(f"GS: rollback 完成, rolled={rolled_streams} pages={len(affected_pages)}")
            if not skip_recompress:
                tlog("GS: rollback libdeflate 开始")
                libdeflate_compress_pdf(cand)
            tlog("GS: rollback save 开始")
            cand.save(
                out_pdf,
                compress_streams=True,
                object_stream_mode=pikepdf.ObjectStreamMode.generate,
            )
            tlog("GS: rollback save 完成")

        return is_valid_pdf(out_pdf), rolled_streams, len(affected_pages)
    except:
        return False, 0, 0
