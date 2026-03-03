"""矢量正则引擎 + run_regex_pass + run_shape_pass"""
import re
import threading

import pikepdf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    CURVE_SIMPLIFY_THRESHOLD, LIBDEFLATE_LEVEL, _vector_engine,
    VECTOR_SPLIT_THRESHOLD_BYTES, VECTOR_CHUNK_TARGET_BYTES,
    VECTOR_INNER_WORKERS, REGEX_STREAM_WORKERS,
)
from utils import safe_print, is_valid_pdf, libdeflate_compress_pdf, tlog


# ─── cm 坐标截断 (翻译矩阵 1 0 0 1 tx ty cm 的 tx/ty 精度) ─────────────────
_NUM_RE_B = rb"[-+]?(?:\d*\.\d+|\d+)"
_CM_RE = re.compile(
    rb"(1\s+0\s+0\s+1\s+)(" + _NUM_RE_B + rb")(\s+)(" + _NUM_RE_B + rb")(\s+cm)"
)


def _fmt_trunc2(b_val):
    try:
        v = float(b_val)
        s = f"{v:.2f}".rstrip("0").rstrip(".")
        if s.startswith("0."):
            s = s[1:]
        elif s.startswith("-0."):
            s = "-" + s[2:]
        elif s == "-0":
            s = "0"
        out = s.encode()
        return out if len(out) < len(b_val) else b_val
    except Exception:
        return b_val


def _apply_cm_truncation(data):
    count = [0]
    def _sub(m):
        nx = _fmt_trunc2(m.group(2))
        ny = _fmt_trunc2(m.group(4))
        if nx != m.group(2) or ny != m.group(4):
            count[0] += 1
        return m.group(1) + nx + m.group(3) + ny + m.group(5)
    return _CM_RE.sub(_sub, data), count[0]


def _optimize_vector_stream(raw_data, sig_figs, enable_smart_c):
    """单引擎执行：Cython 引擎为必需项，不允许回退。"""
    if not hasattr(_vector_engine, "optimize_stream_scan_nogil"):
        raise RuntimeError(
            "vector_hotspot_cython_nogil 缺少 optimize_stream_scan_nogil，"
            "请检查扩展编译/打包流程。"
        )
    return _vector_engine.optimize_stream_scan_nogil(
        raw_data, sig_figs, enable_smart_c, CURVE_SIMPLIFY_THRESHOLD
    )


def _split_and_optimize_large_stream(raw_data, sig_figs, enable_smart_c, progress_callback=None):
    """超大流分块优化：按行聚合成约 1MB 子块，逐块处理并拼回。"""
    if len(raw_data) <= VECTOR_SPLIT_THRESHOLD_BYTES:
        out = _optimize_vector_stream(raw_data, sig_figs, enable_smart_c)
        if progress_callback:
            progress_callback(len(raw_data))
        return out

    lines = raw_data.splitlines(keepends=True)
    # 若几乎没有换行，按空白边界切块（避免错误切断 token）
    if len(lines) <= 1:
        chunks = []
        n = len(raw_data)
        pos = 0
        while pos < n:
            end = min(pos + VECTOR_CHUNK_TARGET_BYTES, n)
            if end < n:
                split = end
                while split > pos and raw_data[split - 1] not in b" \t\r\n":
                    split -= 1
                if split == pos:
                    split = end
                end = split
            chunks.append(raw_data[pos:end])
            pos = end

        out_parts = [b""] * len(chunks)
        max_workers = min(VECTOR_INNER_WORKERS, len(chunks))
        if max_workers <= 1:
            for idx, ch in enumerate(chunks):
                out_parts[idx] = _optimize_vector_stream(ch, sig_figs, enable_smart_c)
                if progress_callback:
                    progress_callback(len(chunks[idx]))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {
                    ex.submit(_optimize_vector_stream, ch, sig_figs, enable_smart_c): idx
                    for idx, ch in enumerate(chunks)
                }
                for f in as_completed(futures):
                    idx = futures[f]
                    out_parts[idx] = f.result()
                    if progress_callback:
                        progress_callback(len(chunks[idx]))
        return b"".join(out_parts)

    chunks = []
    buf = []
    buf_size = 0
    for ln in lines:
        if buf_size + len(ln) > VECTOR_CHUNK_TARGET_BYTES and buf:
            chunks.append(b"".join(buf))
            buf = [ln]
            buf_size = len(ln)
        else:
            buf.append(ln)
            buf_size += len(ln)
    if buf:
        chunks.append(b"".join(buf))

    out_parts = [b""] * len(chunks)
    max_workers = min(VECTOR_INNER_WORKERS, len(chunks))
    if max_workers <= 1:
        for idx, ch in enumerate(chunks):
            out_parts[idx] = _optimize_vector_stream(ch, sig_figs, enable_smart_c)
            if progress_callback:
                progress_callback(len(chunks[idx]))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_optimize_vector_stream, ch, sig_figs, enable_smart_c): idx
                for idx, ch in enumerate(chunks)
            }
            for f in as_completed(futures):
                idx = futures[f]
                out_parts[idx] = f.result()
                if progress_callback:
                    progress_callback(len(chunks[idx]))

    return b"".join(out_parts)


def run_regex_pass(input_path, output_path, sig_figs, enable_smart_c, desc):
    try:
        inline_img_re = re.compile(rb"(^|\\s)BI(\\s|$)")

        # Phase 1: 单次 open，读取全部候选流到内存
        tlog(f"V({desc}): Phase1 pikepdf.open 开始")
        candidate_raw = {}  # xref -> raw_bytes
        with pikepdf.open(input_path) as pdf:
            tlog(f"V({desc}): Phase1 pikepdf.open 完成")
            # 自适应 sig_figs：超大页面需要更高精度以保护细微矢量特征
            _BASE_DIM = 612.0  # 标准 A4 宽度 (pt)
            max_page_dim = 0.0
            for page in pdf.pages:
                mb = page.get("/MediaBox")
                if mb:
                    try:
                        w = abs(float(mb[2]) - float(mb[0]))
                        h = abs(float(mb[3]) - float(mb[1]))
                        max_page_dim = max(max_page_dim, w, h)
                    except Exception:
                        pass
            if max_page_dim > _BASE_DIM:
                import math as _m
                extra_sf = round(_m.log10(max_page_dim / _BASE_DIM))
                if extra_sf > 0:
                    sig_figs = sig_figs + extra_sf
                    safe_print(f"      [自适应] 页面最大尺寸 {max_page_dim:.0f}pt > 标准 {_BASE_DIM:.0f}pt, sig_figs {sig_figs - extra_sf}→{sig_figs}")

            for i, obj in enumerate(pdf.objects):
                if isinstance(obj, pikepdf.Stream):
                    subtype = str(obj.get("/Subtype") or "")
                    if "/Image" not in subtype and "/Font" not in subtype:
                        try:
                            raw = obj.read_bytes()
                            if not inline_img_re.search(raw):
                                candidate_raw[i] = raw
                        except:
                            pass

        if not candidate_raw:
            return False

        total_bytes = sum(len(v) for v in candidate_raw.values())
        total_xrefs = len(candidate_raw)
        tlog(f"V({desc}): Phase1 完成, {total_xrefs} 流, {total_bytes/1024/1024:.1f} MB")

        # Phase 2: 多线程并行 Cython 处理（纯内存，不再打开 PDF）
        tlog(f"V({desc}): Phase2 tqdm+线程池开始")
        all_results = {}
        done_bytes = [0]
        done_streams = [0]
        pbar_lock = threading.Lock()
        stream_workers = min(max(1, REGEX_STREAM_WORKERS), total_xrefs)

        pbar = tqdm(
            total=max(total_bytes, 1),
            desc=f"      🔥 {desc}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        )

        def on_bytes_done(nbytes):
            with pbar_lock:
                remaining = max(total_bytes - done_bytes[0], 0)
                inc = min(max(int(nbytes), 0), remaining)
                if inc > 0:
                    pbar.update(inc)
                    done_bytes[0] += inc

        def _process_one(xref_raw):
            xref, raw_data = xref_raw
            d = _split_and_optimize_large_stream(
                raw_data, sig_figs, enable_smart_c, progress_callback=on_bytes_done
            )
            if len(d) < len(raw_data):
                return (xref, d)
            return None

        with ThreadPoolExecutor(max_workers=stream_workers) as executor:
            futures = {
                executor.submit(_process_one, item): item[0]
                for item in candidate_raw.items()
            }
            for f in as_completed(futures):
                res = f.result()
                done_streams[0] += 1
                if res:
                    all_results[res[0]] = res[1]
                pbar.set_postfix(streams=f"{done_streams[0]}/{total_xrefs}")

        if done_bytes[0] < total_bytes:
            pbar.update(total_bytes - done_bytes[0])
        pbar.close()
        tlog(f"V({desc}): Phase2 tqdm完成")

        # 释放原始数据内存
        candidate_raw.clear()

        if not all_results:
            return False

        # Phase 3: pikepdf 写回 + libdeflate 精压
        tlog(f"V({desc}): Phase3 pikepdf.open 开始")
        with pikepdf.open(input_path) as pdf:
            tlog(f"V({desc}): Phase3 pikepdf.open 完成, 写回 {len(all_results)} 流")
            dirty_xrefs = set(all_results.keys())
            for xref, data in all_results.items():
                pdf.objects[xref].write(data)
            tlog(f"V({desc}): Phase3 写回完成")
            all_results.clear()
            libdeflate_compress_pdf(pdf, only_xrefs=dirty_xrefs)
            tlog(f"V({desc}): Phase3 pikepdf.save 开始")
            pdf.save(
                output_path,
                compress_streams=True,
                object_stream_mode=pikepdf.ObjectStreamMode.generate,
            )
            tlog(f"V({desc}): Phase3 pikepdf.save 完成")
        return is_valid_pdf(output_path)
    except:
        return False


# ─── Shape pass: 4-Bézier 圆→菱形 + 零长度段删除 + cm 截断 ──────────────────

def _optimize_shape_stream(raw_data):
    """单流形状优化: Cython 圆/零段检测 + Python cm 截断。"""
    result, circles, zerosegs = _vector_engine.optimize_shapes_scan_nogil(raw_data)
    result, cm_count = _apply_cm_truncation(result)
    return result, circles, zerosegs, cm_count


def run_shape_pass(input_path, output_path, desc="形状简化"):
    """对 PDF 所有内容流执行形状简化 (圆→菱形)。返回 True/False。"""
    try:
        inline_img_re = re.compile(rb"(^|\s)BI(\s|$)")
        candidates = {}

        tlog(f"S({desc}): pikepdf.open 开始")
        with pikepdf.open(input_path) as pdf:
            tlog(f"S({desc}): pikepdf.open 完成")
            for i, obj in enumerate(pdf.objects):
                if isinstance(obj, pikepdf.Stream):
                    sub = str(obj.get("/Subtype") or "")
                    if "/Image" not in sub and "/Font" not in sub:
                        try:
                            raw = obj.read_bytes()
                            if not inline_img_re.search(raw):
                                candidates[i] = raw
                        except Exception:
                            pass

        if not candidates:
            return False

        total_bytes = sum(len(v) for v in candidates.values())
        total_xrefs = len(candidates)
        tlog(f"S({desc}): 读取完成, {total_xrefs} 流, {total_bytes/1024/1024:.1f} MB")
        tlog(f"S({desc}): tqdm+线程池开始")

        all_results = {}
        done_bytes = [0]
        done_streams = [0]
        pbar_lock = threading.Lock()
        stats = [0, 0, 0]  # circles, zerosegs, cm_trunc

        pbar = tqdm(
            total=max(total_bytes, 1),
            desc=f"      🔷 {desc}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        )

        def _process(item):
            xref, raw = item
            opt, c, z, cm = _optimize_shape_stream(raw)
            return (xref, opt if len(opt) < len(raw) else None, c, z, cm, len(raw))

        workers = min(max(1, REGEX_STREAM_WORKERS), total_xrefs)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_process, it): it[0] for it in candidates.items()}
            for f in as_completed(futs):
                xref, opt, c, z, cm, raw_len = f.result()
                with pbar_lock:
                    stats[0] += c
                    stats[1] += z
                    stats[2] += cm
                    done_streams[0] += 1
                    done_bytes[0] += raw_len
                    pbar.update(raw_len)
                    pbar.set_postfix(
                        streams=f"{done_streams[0]}/{total_xrefs}",
                        circles=stats[0],
                    )
                if opt is not None:
                    all_results[xref] = opt

        pbar.close()
        tlog(f"S({desc}): tqdm完成")
        candidates.clear()

        safe_print(f"      [形状] 圆→菱形: {stats[0]:,}, 零段: {stats[1]:,}, cm截断: {stats[2]:,}")

        if not all_results:
            return False

        tlog(f"S({desc}): Phase3 pikepdf.open 开始")
        with pikepdf.open(input_path) as pdf:
            tlog(f"S({desc}): Phase3 pikepdf.open 完成, 写回 {len(all_results)} 流")
            dirty_xrefs = set(all_results.keys())
            for xref, data in all_results.items():
                pdf.objects[xref].write(data)
            tlog(f"S({desc}): Phase3 写回完成")
            all_results.clear()
            libdeflate_compress_pdf(pdf, only_xrefs=dirty_xrefs)
            tlog(f"S({desc}): Phase3 pikepdf.save 开始")
            pdf.save(
                output_path,
                compress_streams=True,
                object_stream_mode=pikepdf.ObjectStreamMode.generate,
            )
            tlog(f"S({desc}): Phase3 pikepdf.save 完成")
        return is_valid_pdf(output_path)
    except Exception:
        return False
