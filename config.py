"""全局配置常量 + 引擎初始化"""
import sys
import os
import importlib
import subprocess
import multiprocessing
from pathlib import Path
from PIL import Image

# 脚本目录
SCRIPT_DIR = Path(__file__).parent.resolve()

# === Cython 矢量引擎加载 ===

def _bootstrap_vector_engine():
    """加载必需的 Cython 矢量引擎。

    规则：
    - EXE/冻结环境：缺失即直接报错。
    - Python 脚本环境：若未找到扩展，则尝试自动编译一次；仍失败则报错并给出指引。
    """
    module_name = "vector_hotspot_cython_nogil"
    try:
        return importlib.import_module(module_name)
    except Exception as first_exc:
        is_frozen = bool(getattr(sys, "frozen", False))
        build_script = SCRIPT_DIR / "build_cython_vector_hotspot.py"

        if (not is_frozen) and build_script.exists():
            try:
                print("[INIT] 未检测到 Cython 矢量引擎，尝试自动编译...", flush=True)
            except Exception:
                pass
            try:
                subprocess.run(
                    [sys.executable, str(build_script), "build_ext", "--inplace"],
                    cwd=str(SCRIPT_DIR),
                    check=True,
                )
                importlib.invalidate_caches()
                return importlib.import_module(module_name)
            except Exception as build_exc:
                raise RuntimeError(
                    "未能自动编译必需的 Cython 矢量引擎。"
                    "请先执行：uv sync --frozen && uv run python build_cython_vector_hotspot.py build_ext --inplace"
                ) from build_exc

        raise RuntimeError(
            "未能加载必需的矢量加速引擎 vector_hotspot_cython_nogil。"
            "请先编译 Cython 扩展并确保其被正确打包到可执行文件中。"
        ) from first_exc


_vector_engine = _bootstrap_vector_engine()

# === ML Pipeline 可用性检查 ===

_ml_pipeline_available = None  # None=未检查, True=可用, False=不可用

def _safe_warn(msg):
    """config 内部专用的安全打印 (避免对 utils 的循环依赖)"""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('gbk', errors='replace').decode('gbk'))

def check_ml_pipeline_available():
    """检查ML管线是否可用（仅检查模型文件和依赖，不加载模型）"""
    global _ml_pipeline_available
    if _ml_pipeline_available is None:
        try:
            models_dir = SCRIPT_DIR / "models"
            required = [
                "slbr-watermark-detector-fp16.onnx",
                "real-esrgan-x4plus-128-fp16.onnx",
                "NAF-DPM_denoiser-fp16.onnx",
                "NAF-DPM_init_predictor-fp16.onnx",
            ]
            missing = [m for m in required if not (models_dir / m).exists()]
            if missing:
                _safe_warn(f"      [WARN] ML模型缺失: {', '.join(missing)}")
                _ml_pipeline_available = False
            else:
                import ml_pipeline  # noqa: F401 — verify importable
                _ml_pipeline_available = True
        except Exception as e:
            _safe_warn(f"      [WARN] ML管线不可用: {e}")
            _ml_pipeline_available = False
    return _ml_pipeline_available


# === 全局配置 ===

Image.MAX_IMAGE_PIXELS = None  # 禁用DecompressionBomb检查

# 兼容性修复
try:
    import fitz
    PDF_REDACT_IMAGE_REMOVE = fitz.PDF_REDACT_IMAGE_REMOVE
except AttributeError:
    PDF_REDACT_IMAGE_REMOVE = 2

# 基础配置
SIZE_THRESHOLD_MB = 100
MIN_IMAGE_SIZE = 2048
CHUNK_SIZE = 20
CURVE_SIMPLIFY_THRESHOLD = 0.10
CPU_CORES = max(1, multiprocessing.cpu_count() - 1)

# JPEG2000 编码并行策略 (imagecodecs + OpenJPEG)
# numthreads: OpenJPEG 内部编码线程数 (释放 GIL)
# JP2K_WORKERS: ThreadPoolExecutor 并发编码任务数
# 最优组合: workers * numthreads ≈ CPU_CORES, 避免过度订阅
JP2K_THREADS = 4  # 每张图编码使用的 OpenJPEG 内部线程数
JP2K_WORKERS = max(2, CPU_CORES // JP2K_THREADS)  # 并发编码任务数

# 切片/灰度检测配置
GRID_SIZE = 20
GLOBAL_FORCE_MONO_THRESHOLD = 0.98
BLOCK_GRAY_PIXEL_THRESHOLD = 0.15
GRAY_LOWER_BOUND = 50
GRAY_UPPER_BOUND = 220
BINARIZE_THRESHOLD = 180
LIBDEFLATE_LEVEL = 12  # libdeflate max compression level (replaces zlib level 9)
TILE_GRID_ROWS = 5
TILE_GRID_COLS = 5
TILE_CHECK_GRID = 4
COLOR_STD_THRESHOLD = 5.0  # 判定是否为彩色的阈值 (越小越容易被判定为彩色)

# 矢量流分块阈值与块大小
VECTOR_SPLIT_THRESHOLD_BYTES = 1 * 1024 * 1024
VECTOR_CHUNK_TARGET_BYTES = 256 * 1024
VECTOR_INNER_WORKERS = max(1, int(os.environ.get("VECTOR_INNER_WORKERS", "1")))
REGEX_STREAM_WORKERS = max(1, int(os.environ.get("REGEX_STREAM_WORKERS", str(min(os.cpu_count() or 4, 8)))))
