"""矢量正则引擎 + run_regex_pass"""
import re
import threading

import pikepdf
import deflate
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    CURVE_SIMPLIFY_THRESHOLD, LIBDEFLATE_LEVEL, _vector_engine,
    VECTOR_SPLIT_THRESHOLD_BYTES, VECTOR_CHUNK_TARGET_BYTES,
    VECTOR_INNER_WORKERS, REGEX_STREAM_WORKERS,
)
from utils import safe_print, is_valid_pdf, _recompress_streams_libdeflate


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
        # 关键修复：PDF 数值语法不接受科学计数法（如 2e+04）。
        # 这里强制使用十进制定点表示，再去掉末尾无效 0。
        if val.is_integer():
            new_str = "{:.0f}".format(val)
        else:
            # sig_figs 作为小数位上限使用，避免输出指数形式
            new_str = f"{val:.{sig_figs}f}".rstrip("0").rstrip(".")
            # 兜底：若被格式化为空，回退为 0
            if not new_str:
                new_str = "0"

        if new_str.startswith("0."):
            new_str = new_str[1:]
        elif new_str.startswith("-0."):
            new_str = "-" + new_str[2:]
        elif new_str == "-0":
            new_str = "0"
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


def _apply_vector_regex_passes(raw_data, sig_figs, enable_smart_c):
    """对一段内容流执行既有矢量正则优化。"""
    d = L_PATTERN.sub(lambda m: replace_2args(m, sig_figs), raw_data)
    d = VY_PATTERN.sub(lambda m: replace_vy(m, sig_figs), d)
    d = M_PATTERN.sub(lambda m: replace_2args(m, sig_figs), d)
    d = W_PATTERN.sub(lambda m: replace_w(m, sig_figs), d)
    if enable_smart_c:
        d = C_PATTERN.sub(lambda m: replace_c_smart(m, sig_figs), d)
    else:
        d = C_PATTERN.sub(lambda m: replace_c_v4(m, sig_figs), d)
    return d


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
        candidate_raw = {}  # xref -> raw_bytes
        with pikepdf.open(input_path) as pdf:
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

        # Phase 2: 多线程并行 Cython 处理（纯内存，不再打开 PDF）
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

        # 释放原始数据内存
        candidate_raw.clear()

        if not all_results:
            return False

        # Phase 3: 单次 open，写回优化结果 (libdeflate 预压缩)
        with pikepdf.open(input_path) as pdf:
            for xref, data in all_results.items():
                compressed = bytes(deflate.zlib_compress(data, LIBDEFLATE_LEVEL))
                pdf.objects[xref].write(compressed, filter=pikepdf.Name("/FlateDecode"))
            pdf.remove_unreferenced_resources()
            pdf.save(
                output_path,
                compress_streams=True,
                object_stream_mode=pikepdf.ObjectStreamMode.generate,
            )
        _recompress_streams_libdeflate(output_path)
        return is_valid_pdf(output_path)
    except:
        return False
