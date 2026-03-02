"""GhostScript 重构 + surgical_clean"""
import os
import shutil
import subprocess

import pikepdf
from tqdm import tqdm

from utils import is_valid_pdf, _recompress_streams_libdeflate


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
        _recompress_streams_libdeflate(output_path)
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
