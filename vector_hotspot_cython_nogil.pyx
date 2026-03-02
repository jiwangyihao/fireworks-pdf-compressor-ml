# cython: language_level=3
from __future__ import annotations

import re
from libc.stdlib cimport strtod, malloc, free
from libc.stdio cimport snprintf
from libc.math cimport floor, log10, fabs, pow as cpow, sqrt as csqrt
from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.string cimport memcpy

CURVE_SIMPLIFY_THRESHOLD = 0.10

NUM_RE = rb"[-+]?(?:\d*\.\d+|\d+)"
L_PATTERN = re.compile(rb"(" + NUM_RE + rb")(\s+)(" + NUM_RE + rb")(\s+)(l)(?=\s|$)")
C_PATTERN = re.compile(
    rb"(" + NUM_RE + rb")(\s+)(" + NUM_RE + rb")(\s+)"
    + rb"(" + NUM_RE + rb")(\s+)(" + NUM_RE + rb")(\s+)"
    + rb"(" + NUM_RE + rb")(\s+)(" + NUM_RE + rb")(\s+)(c)(?=\s|$)"
)
VY_PATTERN = re.compile(
    rb"(" + NUM_RE + rb")(\s+)(" + NUM_RE + rb")(\s+)"
    + rb"(" + NUM_RE + rb")(\s+)(" + NUM_RE + rb")(\s+)([vy])(?=\s|$)"
)
M_PATTERN = re.compile(rb"(" + NUM_RE + rb")(\s+)(" + NUM_RE + rb")(\s+)(m)(?=\s|$)")
W_PATTERN = re.compile(rb"(" + NUM_RE + rb")(\s+)(w)(?=\s|$)")

cdef bytes _format_number_nogil(bytes bytes_val, int sig_figs):
    cdef Py_ssize_t n = len(bytes_val)
    cdef const char* src
    cdef char* endptr
    cdef double val
    cdef char* outbuf = NULL
    cdef int written = 0
    cdef bytes out
    cdef bytes trimmed

    if n < 3:
        return bytes_val

    src = bytes_val
    outbuf = <char*>malloc(128)
    if outbuf == NULL:
        return bytes_val

    with nogil:
        val = strtod(src, &endptr)

    if endptr == src:
        free(outbuf)
        return bytes_val

    if floor(val) == val:
        with nogil:
            written = snprintf(outbuf, 128, "%.0f", val)
    else:
        with nogil:
            written = snprintf(outbuf, 128, "%.*f", sig_figs, val)

    if written <= 0 or written >= 128:
        free(outbuf)
        return bytes_val

    out = (<bytes>outbuf[:written])
    free(outbuf)

    trimmed = out.rstrip(b"0").rstrip(b".") if b"." in out else out
    if len(trimmed) == 0:
        trimmed = b"0"
    if trimmed.startswith(b"0."):
        trimmed = trimmed[1:]
    elif trimmed.startswith(b"-0."):
        trimmed = b"-" + trimmed[2:]
    elif trimmed == b"-0":
        trimmed = b"0"

    if len(trimmed) < n:
        return trimmed
    return bytes_val


def _r2(m, int sig_figs):
    return (
        _format_number_nogil(m.group(1), sig_figs)
        + m.group(2)
        + _format_number_nogil(m.group(3), sig_figs)
        + m.group(4)
        + m.group(5)
    )


def _rw(m, int sig_figs):
    return _format_number_nogil(m.group(1), sig_figs) + m.group(2) + m.group(3)


def _rvy(m, int sig_figs):
    n1 = _format_number_nogil(m.group(1), sig_figs)
    n2 = _format_number_nogil(m.group(3), sig_figs)
    n3 = _format_number_nogil(m.group(5), sig_figs)
    n4 = _format_number_nogil(m.group(7), sig_figs)
    op = m.group(9)
    return n1 + m.group(2) + n2 + m.group(4) + n3 + m.group(6) + n4 + m.group(8) + op


def _rc4(m, int sig_figs):
    nums = [_format_number_nogil(m.group(i), sig_figs) for i in range(1, 12, 2)]
    spaces = [m.group(i) for i in range(2, 13, 2)]
    return b"".join(n + s for n, s in zip(nums, spaces)) + m.group(13)


def optimize_stream_nogil(bytes raw_data, int sig_figs=4, bint enable_smart_c=False):
    cdef bytes d
    d = L_PATTERN.sub(lambda m: _r2(m, sig_figs), raw_data)
    d = VY_PATTERN.sub(lambda m: _rvy(m, sig_figs), d)
    d = M_PATTERN.sub(lambda m: _r2(m, sig_figs), d)
    d = W_PATTERN.sub(lambda m: _rw(m, sig_figs), d)
    d = C_PATTERN.sub(lambda m: _rc4(m, sig_figs), d)
    return d


cdef inline bint _is_digit(char c) nogil:
    return c >= c'0' and c <= c'9'


cdef inline bint _is_space(char c) nogil:
    return c == c' ' or c == c'\t' or c == c'\r' or c == c'\n' or c == c'\f' or c == c'\v'


cdef inline bint _looks_number_start(const char* s, Py_ssize_t i, Py_ssize_t n) nogil:
    cdef char c = s[i]
    if _is_digit(c):
        return True
    if c == c'.':
        return (i + 1 < n and _is_digit(s[i + 1]))
    if c == c'+' or c == c'-':
        if i + 1 >= n:
            return False
        return _is_digit(s[i + 1]) or s[i + 1] == c'.'
    return False


cdef inline Py_ssize_t _parse_num_end(const char* s, Py_ssize_t i, Py_ssize_t n) nogil:
    cdef Py_ssize_t j = i
    if j < n and (s[j] == c'+' or s[j] == c'-'):
        j += 1
    while j < n:
        if _is_digit(s[j]) or s[j] == c'.':
            j += 1
            continue
        if (s[j] == c'+' or s[j] == c'-') and j > i and (s[j - 1] == c'e' or s[j - 1] == c'E'):
            j += 1
            continue
        if s[j] == c'e' or s[j] == c'E':
            if j + 1 < n and (_is_digit(s[j + 1]) or s[j + 1] == c'+' or s[j + 1] == c'-'):
                j += 1
                continue
        break
    return j


cdef inline Py_ssize_t _skip_ws(const char* s, Py_ssize_t i, Py_ssize_t n) nogil:
    cdef Py_ssize_t j = i
    while j < n and _is_space(s[j]):
        j += 1
    return j


cdef inline int _format_shorter(const char* s, Py_ssize_t st, Py_ssize_t ed, int sig_figs, char* outbuf) nogil:
    cdef char numbuf[128]
    cdef char* endptr
    cdef double val, av, scale
    cdef int out_len
    cdef int k
    cdef int dec_places
    cdef int leading
    cdef int tok_len = <int>(ed - st)
    cdef bint has_dot

    if tok_len <= 0 or tok_len >= 120:
        return 0

    memcpy(numbuf, s + st, tok_len)
    numbuf[tok_len] = c'\0'

    val = strtod(numbuf, &endptr)
    if endptr == numbuf:
        return 0

    if floor(val) == val:
        # Integer value: round to sig_figs significant figures
        av = fabs(val)
        if av >= 1.0:
            leading = <int>floor(log10(av)) + 1
            if leading > sig_figs:
                scale = cpow(10.0, <double>(leading - sig_figs))
                val = floor(val / scale + 0.5) * scale
        out_len = snprintf(outbuf, 128, "%.0f", val)
    else:
        # Non-integer: compute decimal places for sig_figs significant figures
        av = fabs(val)
        if av >= 1.0:
            leading = <int>floor(log10(av)) + 1
            dec_places = sig_figs - leading
        else:
            dec_places = <int>(-floor(log10(av))) - 1 + sig_figs
        # When dec_places < 0, round off integer digits first
        if dec_places < 0:
            scale = cpow(10.0, <double>(-dec_places))
            val = floor(val / scale + 0.5) * scale
            dec_places = 0
        out_len = snprintf(outbuf, 128, "%.*f", dec_places, val)

    if out_len <= 0 or out_len >= 120:
        return 0

    # Strip trailing zeros only when there is a decimal point
    has_dot = False
    for k in range(out_len):
        if outbuf[k] == c'.':
            has_dot = True
            break
    if has_dot:
        k = out_len - 1
        while k >= 0 and outbuf[k] == c'0':
            k -= 1
        if k >= 0 and outbuf[k] == c'.':
            k -= 1
        out_len = k + 1
        if out_len <= 0:
            outbuf[0] = c'0'
            out_len = 1

    # Leading zero compression: "0.xxx" -> ".xxx"
    if out_len >= 2 and outbuf[0] == c'0' and outbuf[1] == c'.':
        memcpy(outbuf, outbuf + 1, out_len - 1)
        out_len -= 1
    elif out_len >= 3 and outbuf[0] == c'-' and outbuf[1] == c'0' and outbuf[2] == c'.':
        memcpy(outbuf + 1, outbuf + 2, out_len - 2)
        out_len -= 1
    elif out_len == 2 and outbuf[0] == c'-' and outbuf[1] == c'0':
        outbuf[0] = c'0'
        out_len = 1

    if out_len < tok_len:
        return out_len
    return 0


cdef inline bint _match_cmd(const char* s, Py_ssize_t i, Py_ssize_t n, int k,
                            char op1, char op2,
                            Py_ssize_t* ns, Py_ssize_t* ne,
                            Py_ssize_t* ws_s, Py_ssize_t* ws_e,
                            Py_ssize_t* op_pos) nogil:
    cdef int idx
    cdef Py_ssize_t p = i
    cdef Py_ssize_t q
    cdef Py_ssize_t w0
    cdef char op

    for idx in range(k):
        if not _looks_number_start(s, p, n):
            return False
        q = _parse_num_end(s, p, n)
        if q <= p:
            return False
        ns[idx] = p
        ne[idx] = q
        p = q
        if idx < k - 1:
            w0 = p
            p = _skip_ws(s, p, n)
            if p == w0:
                return False
            ws_s[idx] = w0
            ws_e[idx] = p

    w0 = p
    p = _skip_ws(s, p, n)
    if p == w0:
        return False
    ws_s[k - 1] = w0
    ws_e[k - 1] = p

    if p >= n:
        return False
    op = s[p]
    if not (op == op1 or op == op2):
        return False
    if p + 1 < n and (not _is_space(s[p + 1])):
        return False
    op_pos[0] = p
    return True


cdef Py_ssize_t _scan_replace_core(const char* src, Py_ssize_t n, int sig_figs, char* dst) nogil:
    cdef char numbuf[128]
    cdef char outbuf[128]
    cdef char* endptr
    cdef double val
    cdef int out_len
    cdef int k
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j
    cdef Py_ssize_t tok_len
    cdef Py_ssize_t w = 0
    cdef bint is_int

    while i < n:
        if not _looks_number_start(src, i, n):
            dst[w] = src[i]
            w += 1
            i += 1
            continue

        j = i
        if j < n and (src[j] == c'+' or src[j] == c'-'):
            j += 1

        while j < n:
            if _is_digit(src[j]) or src[j] == c'.':
                j += 1
                continue
            if (src[j] == c'+' or src[j] == c'-') and j > i and (src[j - 1] == c'e' or src[j - 1] == c'E'):
                j += 1
                continue
            if src[j] == c'e' or src[j] == c'E':
                if j + 1 < n and (_is_digit(src[j + 1]) or src[j + 1] == c'+' or src[j + 1] == c'-'):
                    j += 1
                    continue
            break

        tok_len = j - i
        if tok_len <= 0:
            dst[w] = src[i]
            w += 1
            i += 1
            continue

        if tok_len >= 120:
            memcpy(dst + w, src + i, tok_len)
            w += tok_len
            i = j
            continue

        memcpy(numbuf, src + i, tok_len)
        numbuf[tok_len] = c'\0'

        val = strtod(numbuf, &endptr)
        if endptr == numbuf:
            memcpy(dst + w, src + i, tok_len)
            w += tok_len
            i = j
            continue

        is_int = (floor(val) == val)
        if is_int:
            out_len = snprintf(outbuf, 128, "%.0f", val)
        else:
            out_len = snprintf(outbuf, 128, "%.*f", sig_figs, val)

        if out_len <= 0 or out_len >= 120:
            memcpy(dst + w, src + i, tok_len)
            w += tok_len
            i = j
            continue

        if not is_int:
            k = out_len - 1
            while k >= 0 and outbuf[k] == c'0':
                k -= 1
            if k >= 0 and outbuf[k] == c'.':
                k -= 1
            out_len = k + 1
            if out_len <= 0:
                outbuf[0] = c'0'
                out_len = 1

        if out_len >= 2 and outbuf[0] == c'0' and outbuf[1] == c'.':
            memcpy(outbuf, outbuf + 1, out_len - 1)
            out_len -= 1
        elif out_len >= 3 and outbuf[0] == c'-' and outbuf[1] == c'0' and outbuf[2] == c'.':
            memcpy(outbuf + 1, outbuf + 2, out_len - 2)
            out_len -= 1
        elif out_len == 2 and outbuf[0] == c'-' and outbuf[1] == c'0':
            outbuf[0] = c'0'
            out_len = 1

        if out_len < tok_len:
            memcpy(dst + w, outbuf, out_len)
            w += out_len
        else:
            memcpy(dst + w, src + i, tok_len)
            w += tok_len

        i = j

    return w


cdef inline double _parse_double(const char* src, Py_ssize_t start, Py_ssize_t end, char* buf) nogil:
    """Parse a double from src[start:end] using a temp buffer for null-termination."""
    cdef Py_ssize_t length = end - start
    cdef char* endptr
    if length <= 0 or length >= 127:
        return 0.0
    memcpy(buf, src + start, length)
    buf[length] = c'\0'
    return strtod(buf, &endptr)


def optimize_stream_scan_nogil(bytes raw_data, int sig_figs=4, bint enable_smart_c=False, double curve_threshold=0.10):
    cdef Py_ssize_t n = len(raw_data)
    cdef const char* src = raw_data
    cdef char* dst = <char*>malloc(n + 1)
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t w = 0
    cdef Py_ssize_t ns[6]
    cdef Py_ssize_t ne[6]
    cdef Py_ssize_t ws_s[6]
    cdef Py_ssize_t ws_e[6]
    cdef Py_ssize_t op_pos[1]
    cdef int out_len
    cdef int idx
    cdef int k
    cdef char fmtbuf[128]
    cdef char numbuf_tmp[128]
    cdef bytes out
    # 曲线简化: 当前点追踪
    cdef double cur_x = 0.0, cur_y = 0.0
    cdef double subpath_x = 0.0, subpath_y = 0.0
    cdef double cp1x, cp1y, cp2x, cp2y, epx, epy
    cdef double dx, dy, line_len_sq, line_len, cross1, cross2, dist1, dist2, max_dist, rel_dist
    cdef double REL_THRESHOLD = 0.10

    if dst == NULL:
        return raw_data

    with nogil:
        while i < n:
            if not _looks_number_start(src, i, n):
                # h (closepath) 重置当前点到子路径起点
                if enable_smart_c and src[i] == c'h':
                    cur_x = subpath_x
                    cur_y = subpath_y
                dst[w] = src[i]
                w += 1
                i += 1
                continue

            # === c 命令 (6 参数) ===
            if _match_cmd(src, i, n, 6, c'c', c'c', &ns[0], &ne[0], &ws_s[0], &ws_e[0], &op_pos[0]):
                if enable_smart_c:
                    cp1x = _parse_double(src, ns[0], ne[0], numbuf_tmp)
                    cp1y = _parse_double(src, ns[1], ne[1], numbuf_tmp)
                    cp2x = _parse_double(src, ns[2], ne[2], numbuf_tmp)
                    cp2y = _parse_double(src, ns[3], ne[3], numbuf_tmp)
                    epx  = _parse_double(src, ns[4], ne[4], numbuf_tmp)
                    epy  = _parse_double(src, ns[5], ne[5], numbuf_tmp)

                    # 计算控制点到弦 (cur→end) 的最大距离
                    dx = epx - cur_x
                    dy = epy - cur_y
                    line_len_sq = dx * dx + dy * dy
                    if line_len_sq < 1e-12:
                        dist1 = csqrt((cp1x - cur_x) * (cp1x - cur_x) + (cp1y - cur_y) * (cp1y - cur_y))
                        dist2 = csqrt((cp2x - cur_x) * (cp2x - cur_x) + (cp2y - cur_y) * (cp2y - cur_y))
                        max_dist = dist1 if dist1 > dist2 else dist2
                        rel_dist = 999.0  # 零长弦不简化
                    else:
                        line_len = csqrt(line_len_sq)
                        cross1 = (cp1x - cur_x) * dy - (cp1y - cur_y) * dx
                        cross2 = (cp2x - cur_x) * dy - (cp2y - cur_y) * dx
                        dist1 = fabs(cross1) / line_len
                        dist2 = fabs(cross2) / line_len
                        max_dist = dist1 if dist1 > dist2 else dist2
                        rel_dist = max_dist / line_len

                    if max_dist < curve_threshold and rel_dist < REL_THRESHOLD:
                        # c → l 简化: 只输出 x3 y3 l
                        out_len = _format_shorter(src, ns[4], ne[4], sig_figs, fmtbuf)
                        if out_len > 0:
                            memcpy(dst + w, fmtbuf, out_len)
                            w += out_len
                        else:
                            memcpy(dst + w, src + ns[4], ne[4] - ns[4])
                            w += (ne[4] - ns[4])
                        dst[w] = c' '
                        w += 1
                        out_len = _format_shorter(src, ns[5], ne[5], sig_figs, fmtbuf)
                        if out_len > 0:
                            memcpy(dst + w, fmtbuf, out_len)
                            w += out_len
                        else:
                            memcpy(dst + w, src + ns[5], ne[5] - ns[5])
                            w += (ne[5] - ns[5])
                        dst[w] = c' '
                        w += 1
                        dst[w] = c'l'
                        w += 1
                        cur_x = epx
                        cur_y = epy
                        i = op_pos[0] + 1
                        continue
                    # 未简化, 更新当前点后正常输出
                    cur_x = epx
                    cur_y = epy
                # 正常输出 c 命令的 6 个参数
                for idx in range(6):
                    out_len = _format_shorter(src, ns[idx], ne[idx], sig_figs, fmtbuf)
                    if out_len > 0:
                        memcpy(dst + w, fmtbuf, out_len)
                        w += out_len
                    else:
                        memcpy(dst + w, src + ns[idx], ne[idx] - ns[idx])
                        w += (ne[idx] - ns[idx])
                    memcpy(dst + w, src + ws_s[idx], ws_e[idx] - ws_s[idx])
                    w += (ws_e[idx] - ws_s[idx])
                dst[w] = src[op_pos[0]]
                w += 1
                i = op_pos[0] + 1
                continue

            # === v/y 命令 (4 参数) ===
            if _match_cmd(src, i, n, 4, c'v', c'y', &ns[0], &ne[0], &ws_s[0], &ws_e[0], &op_pos[0]):
                k = 4
            # === m/l 命令 (2 参数) ===
            elif _match_cmd(src, i, n, 2, c'm', c'l', &ns[0], &ne[0], &ws_s[0], &ws_e[0], &op_pos[0]):
                k = 2
            # === w 命令 (1 参数) ===
            elif _match_cmd(src, i, n, 1, c'w', c'w', &ns[0], &ne[0], &ws_s[0], &ws_e[0], &op_pos[0]):
                k = 1
            else:
                ne[0] = _parse_num_end(src, i, n)
                memcpy(dst + w, src + i, ne[0] - i)
                w += (ne[0] - i)
                i = ne[0]
                continue

            # 正常输出 v/y, m/l, w 命令
            for idx in range(k):
                out_len = _format_shorter(src, ns[idx], ne[idx], sig_figs, fmtbuf)
                if out_len > 0:
                    memcpy(dst + w, fmtbuf, out_len)
                    w += out_len
                else:
                    memcpy(dst + w, src + ns[idx], ne[idx] - ns[idx])
                    w += (ne[idx] - ns[idx])
                memcpy(dst + w, src + ws_s[idx], ws_e[idx] - ws_s[idx])
                w += (ws_e[idx] - ws_s[idx])
            dst[w] = src[op_pos[0]]
            w += 1

            # 追踪当前点 (m/l: 最后2参数, v/y: 最后2参数)
            if enable_smart_c and k >= 2:
                cur_x = _parse_double(src, ns[k - 2], ne[k - 2], numbuf_tmp)
                cur_y = _parse_double(src, ns[k - 1], ne[k - 1], numbuf_tmp)
                if k == 2 and src[op_pos[0]] == c'm':
                    subpath_x = cur_x
                    subpath_y = cur_y

            i = op_pos[0] + 1

    out = <bytes>PyBytes_FromStringAndSize(dst, w)
    free(dst)
    return out


def optimize_stream_scan_strict(bytes raw_data, int sig_figs=4):
    """严格正确优先版：
    - 仅在 PDF 内容流的运算符上下文中压缩数值；
    - 跳过注释、字面字符串、十六进制字符串、名称对象；
    - 语义目标贴近原 re 规则（m/l/w/v/y/c 前导操作数）。
    """
    cdef Py_ssize_t n = len(raw_data)
    cdef Py_ssize_t i = 0
    cdef int depth_paren = 0
    cdef bint in_comment = False
    cdef bint in_hex = False
    cdef bint escaped = False

    tokens = []      # [(kind, raw_bytes)] kind in {'num','op','other'}
    seps = []        # separator bytes between tokens
    cur_sep = bytearray()

    def flush_sep():
        nonlocal cur_sep
        seps.append(bytes(cur_sep))
        cur_sep.clear()

    while i < n:
        c = raw_data[i]

        # 注释态
        if in_comment:
            cur_sep.append(c)
            if c == 10 or c == 13:
                in_comment = False
            i += 1
            continue

        # 字面字符串态 (...)
        if depth_paren > 0:
            cur_sep.append(c)
            if escaped:
                escaped = False
            elif c == 92:  # '\\'
                escaped = True
            elif c == 40:  # '('
                depth_paren += 1
            elif c == 41:  # ')'
                depth_paren -= 1
            i += 1
            continue

        # 十六进制字符串态 <...>
        if in_hex:
            cur_sep.append(c)
            if c == 62:  # '>'
                in_hex = False
            i += 1
            continue

        # 态切换入口
        if c == 37:  # '%'
            in_comment = True
            cur_sep.append(c)
            i += 1
            continue
        if c == 40:  # '('
            depth_paren = 1
            cur_sep.append(c)
            i += 1
            continue
        if c == 60:  # '<'
            if i + 1 < n and raw_data[i + 1] == 60:  # <<
                cur_sep.extend(b"<<")
                i += 2
                continue
            in_hex = True
            cur_sep.append(c)
            i += 1
            continue
        if c == 62 and i + 1 < n and raw_data[i + 1] == 62:  # >>
            cur_sep.extend(b">>")
            i += 2
            continue

        # 空白或分隔符
        if c in b" \t\r\n\f\v[]{}":
            cur_sep.append(c)
            i += 1
            continue

        # 名称对象 /Name：整体视为 other，不压缩内部数字
        if c == 47:  # '/'
            flush_sep()
            j = i + 1
            while j < n and raw_data[j] not in b" \t\r\n\f\v[]{}()<>/%":
                j += 1
            tokens.append(("other", raw_data[i:j]))
            i = j
            continue

        # 读取常规 token
        flush_sep()
        j = i
        while j < n and raw_data[j] not in b" \t\r\n\f\v[]{}()<>/%":
            j += 1
        tok = raw_data[i:j]

        # 判定 number token
        is_num = True
        try:
            _ = float(tok)
        except Exception:
            is_num = False

        if is_num:
            tokens.append(("num", tok))
        elif len(tok) == 1 and tok in (b"m", b"l", b"w", b"v", b"y", b"c"):
            tokens.append(("op", tok))
        else:
            tokens.append(("other", tok))
        i = j

    # 尾分隔
    seps.append(bytes(cur_sep))

    # 基于运算符上下文替换（仅替换必要的前导数值）
    values = [v for _, v in tokens]
    kinds = [k for k, _ in tokens]

    def fmt_num(b):
        return _format_number_nogil(b, sig_figs)

    for idx, (k, v) in enumerate(tokens):
        if k != "op":
            continue
        if v == b"w":
            if idx - 1 >= 0 and kinds[idx - 1] == "num":
                values[idx - 1] = fmt_num(values[idx - 1])
        elif v in (b"m", b"l"):
            if idx - 2 >= 0 and kinds[idx - 2] == "num" and kinds[idx - 1] == "num":
                values[idx - 2] = fmt_num(values[idx - 2])
                values[idx - 1] = fmt_num(values[idx - 1])
        elif v in (b"v", b"y"):
            if idx - 4 >= 0 and all(kinds[idx - t] == "num" for t in (4,3,2,1)):
                values[idx - 4] = fmt_num(values[idx - 4])
                values[idx - 3] = fmt_num(values[idx - 3])
                values[idx - 2] = fmt_num(values[idx - 2])
                values[idx - 1] = fmt_num(values[idx - 1])
        elif v == b"c":
            if idx - 6 >= 0 and all(kinds[idx - t] == "num" for t in (6,5,4,3,2,1)):
                for t in (6,5,4,3,2,1):
                    values[idx - t] = fmt_num(values[idx - t])

    # 重组：sep0 tok0 sep1 tok1 ... sepN
    out = bytearray()
    for i in range(len(tokens)):
        out.extend(seps[i])
        out.extend(values[i])
    out.extend(seps[len(tokens)])
    return bytes(out)
