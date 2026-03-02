"""通用工具函数 (无业务逻辑依赖)"""
import os
import zlib
import ctypes
from concurrent.futures import ThreadPoolExecutor

import pikepdf
from tqdm import tqdm

from config import LIBDEFLATE_LEVEL


# ============================
# ctypes libdeflate (释放 GIL，支持多线程并发)
# ============================

def _find_libdeflate_dll():
    """查找 libdeflate DLL，按优先级：项目内 -> conda 环境 -> 系统路径。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "deflate.dll"),
        os.path.join(script_dir, "libdeflate.dll"),
    ]
    # conda 环境
    import sys
    conda_bin = os.path.join(os.path.dirname(sys.executable), "Library", "bin")
    candidates.append(os.path.join(conda_bin, "deflate.dll"))
    candidates.append(os.path.join(conda_bin, "libdeflate.dll"))
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


_libdeflate_dll = None
_CTYPES_AVAILABLE = False


def _init_ctypes_libdeflate():
    """延迟初始化 ctypes libdeflate。"""
    global _libdeflate_dll, _CTYPES_AVAILABLE
    if _libdeflate_dll is not None:
        return _CTYPES_AVAILABLE
    dll_path = _find_libdeflate_dll()
    if dll_path is None:
        _libdeflate_dll = False
        _CTYPES_AVAILABLE = False
        return False
    try:
        lib = ctypes.CDLL(dll_path)
        lib.libdeflate_alloc_compressor.restype = ctypes.c_void_p
        lib.libdeflate_alloc_compressor.argtypes = [ctypes.c_int]
        lib.libdeflate_zlib_compress.restype = ctypes.c_size_t
        lib.libdeflate_zlib_compress.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
        ]
        lib.libdeflate_free_compressor.restype = None
        lib.libdeflate_free_compressor.argtypes = [ctypes.c_void_p]
        _libdeflate_dll = lib
        _CTYPES_AVAILABLE = True
        return True
    except Exception:
        _libdeflate_dll = False
        _CTYPES_AVAILABLE = False
        return False


def _ctypes_zlib_compress(data, level=LIBDEFLATE_LEVEL):
    """通过 ctypes 调用 libdeflate (释放 GIL)。"""
    compressor = _libdeflate_dll.libdeflate_alloc_compressor(level)
    if not compressor:
        raise MemoryError
    try:
        in_len = len(data)
        out_size = in_len + 256 + in_len // 100
        out_buf = ctypes.create_string_buffer(out_size)
        result_size = _libdeflate_dll.libdeflate_zlib_compress(
            compressor, data, in_len, out_buf, out_size,
        )
        if result_size == 0:
            raise RuntimeError
        return bytes(out_buf[:result_size])
    finally:
        _libdeflate_dll.libdeflate_free_compressor(compressor)


def _compress_one_stream(args):
    """单个流的压缩任务（在线程池中执行）。"""
    raw, old_size, was_flate, level = args
    if was_flate:
        new_compressed = _ctypes_zlib_compress(raw, level)
        if len(new_compressed) < old_size:
            return new_compressed, old_size - len(new_compressed)
    else:
        if len(raw) < 64:
            return None, 0
        new_compressed = _ctypes_zlib_compress(raw, level)
        if len(new_compressed) < len(raw):
            return new_compressed, len(raw) - len(new_compressed)
    return None, 0


def zlib_compress(data, level=LIBDEFLATE_LEVEL):
    """统一压缩接口：ctypes libdeflate > stdlib zlib。"""
    if _init_ctypes_libdeflate():
        return _ctypes_zlib_compress(data, level)
    return zlib.compress(data, min(level, 9))


def safe_print(msg):
    """安全打印，处理 Windows 控制台的编码问题"""
    try:
        print(msg)
    except UnicodeEncodeError:
        # 移除无法编码的字符 (主要是 emoji)
        safe_msg = msg.encode('gbk', errors='replace').decode('gbk')
        print(safe_msg)


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


_SKIP_FILTERS = {
    pikepdf.Name("/DCTDecode"), pikepdf.Name("/JPXDecode"),
    pikepdf.Name("/CCITTFaxDecode"), pikepdf.Name("/JBIG2Decode"),
    pikepdf.Name("/LZWDecode"), pikepdf.Name("/Crypt"),
}


def _get_num_workers():
    """获取线程池工作线程数。"""
    try:
        return max(1, min(os.cpu_count() or 1, 8))
    except Exception:
        return 1


def libdeflate_compress_pdf(pdf, level=LIBDEFLATE_LEVEL):
    """对已打开的 pikepdf PDF 对象做 libdeflate 全流压缩（in-memory，不写磁盘）。

    在 pikepdf.save() 之前调用，避免双重压缩开销。
    ctypes DLL 可用时自动启用多线程并发（释放 GIL），否则回退到单线程 deflate 包。
    """
    candidates = []
    for obj in pdf.objects:
        if not isinstance(obj, pikepdf.Stream):
            continue
        filt = obj.get("/Filter")
        if isinstance(filt, pikepdf.Array):
            continue
        if filt in _SKIP_FILTERS:
            continue
        if filt == pikepdf.Name("/FlateDecode") or filt is None:
            candidates.append(obj)
    if not candidates:
        return

    use_parallel = _init_ctypes_libdeflate() and len(candidates) >= 8

    if use_parallel:
        _libdeflate_compress_parallel(candidates, level)
    else:
        _libdeflate_compress_sequential(candidates, level)


def _libdeflate_compress_parallel(candidates, level):
    """多线程并发压缩（ctypes libdeflate，释放 GIL）。"""
    # Phase 1: 读取所有流的原始数据（必须在主线程、pikepdf 不是线程安全的）
    tasks = []
    for obj in candidates:
        filt = obj.get("/Filter")
        try:
            if filt == pikepdf.Name("/FlateDecode"):
                raw = obj.read_bytes()
                old_size = len(obj.read_raw_bytes())
                tasks.append((raw, old_size, True, level))
            elif filt is None:
                raw = obj.read_bytes()
                tasks.append((raw, len(raw), False, level))
            else:
                tasks.append(None)
        except Exception:
            tasks.append(None)

    # Phase 2: 多线程并发压缩（释放 GIL）
    num_workers = _get_num_workers()
    valid_tasks = [(i, t) for i, t in enumerate(tasks) if t is not None]
    results_map = {}

    pbar = tqdm(total=len(valid_tasks), desc=f"      🔧 libdeflate x{num_workers}T", unit="stream", leave=True)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_compress_one_stream, t): i for i, t in valid_tasks}
        improved = 0
        saved_bytes = 0
        for future in futures:
            idx = futures[future]
            try:
                compressed, saving = future.result()
                if compressed is not None:
                    results_map[idx] = compressed
                    improved += 1
                    saved_bytes += saving
            except Exception:
                pass
            pbar.update(1)
            pbar.set_postfix(improved=improved, saved=f"{saved_bytes/1024:.0f}KB")

    pbar.close()

    # Phase 3: 写回压缩结果（主线程）
    for idx, compressed in results_map.items():
        candidates[idx].write(compressed, filter=pikepdf.Name("/FlateDecode"))


def _libdeflate_compress_sequential(candidates, level):
    """单线程压缩回退（使用 deflate 包）。"""
    improved = 0
    saved_bytes = 0
    pbar = tqdm(candidates, desc="      🔧 libdeflate", unit="stream", leave=True)
    for obj in pbar:
        filt = obj.get("/Filter")
        try:
            if filt == pikepdf.Name("/FlateDecode"):
                raw = obj.read_bytes()
                old_size = len(obj.read_raw_bytes())
                new_compressed = zlib_compress(raw, level)
                if len(new_compressed) < old_size:
                    obj.write(new_compressed, filter=pikepdf.Name("/FlateDecode"))
                    improved += 1
                    saved_bytes += old_size - len(new_compressed)
            elif filt is None:
                raw = obj.read_bytes()
                if len(raw) < 64:
                    continue
                new_compressed = zlib_compress(raw, level)
                if len(new_compressed) < len(raw):
                    obj.write(new_compressed, filter=pikepdf.Name("/FlateDecode"))
                    improved += 1
                    saved_bytes += len(raw) - len(new_compressed)
        except Exception:
            continue
        pbar.set_postfix(improved=improved, saved=f"{saved_bytes/1024:.0f}KB")
    pbar.close()


def _recompress_streams_libdeflate(pdf_path, level=LIBDEFLATE_LEVEL):
    """文件级 libdeflate 重压缩（用于 GS/fitz 等外部保存后的后处理）。"""
    try:
        with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
            libdeflate_compress_pdf(pdf, level)
            pdf.save(pdf_path)
    except Exception:
        pass
