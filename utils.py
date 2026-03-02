"""通用工具函数 (无业务逻辑依赖)"""
import os

import pikepdf
import deflate
from tqdm import tqdm

from config import LIBDEFLATE_LEVEL


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


def libdeflate_compress_pdf(pdf, level=LIBDEFLATE_LEVEL):
    """对已打开的 pikepdf PDF 对象做 libdeflate 全流压缩（in-memory，不写磁盘）。

    在 pikepdf.save() 之前调用，避免双重压缩开销。
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
    improved = 0
    saved_bytes = 0
    pbar = tqdm(candidates, desc="      🔧 libdeflate", unit="stream", leave=True)
    for obj in pbar:
        filt = obj.get("/Filter")
        try:
            if filt == pikepdf.Name("/FlateDecode"):
                raw = obj.read_bytes()
                old_size = len(obj.read_raw_bytes())
                new_compressed = bytes(deflate.zlib_compress(raw, level))
                if len(new_compressed) < old_size:
                    obj.write(new_compressed, filter=pikepdf.Name("/FlateDecode"))
                    improved += 1
                    saved_bytes += old_size - len(new_compressed)
            elif filt is None:
                raw = obj.read_bytes()
                if len(raw) < 64:
                    continue
                new_compressed = bytes(deflate.zlib_compress(raw, level))
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
