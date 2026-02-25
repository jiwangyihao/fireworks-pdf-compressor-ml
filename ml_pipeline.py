"""
5-Stage Pipelined ML Document Enhancement

Producer-consumer pipeline:
    Render -> [Q0] -> SLBR -> [Q1] -> ESRGAN -> [Q2] -> NAF-DPM -> [Q3] -> DoxaPy -> results

Architecture:
    - Render:  Background thread, PDF page rasterization via PyMuPDF
               Max 5 pages ahead of SLBR
    - SLBR:    CPU thread pool (3-4 workers, each with own ORT CPU session)
               GPU fallback if CPU can't keep up (runs on main thread)
               Max 5 pages ahead of ESRGAN
    - ESRGAN:  GPU on main thread, tiled with IO binding
    - NAF-DPM: GPU on main thread, cross-page batch packing for full utilization
    - DoxaPy:  CPU thread pool (1-2 workers)

Constraints:
    - All GPU work runs on the SAME thread (ORT DML not thread-safe)
    - ESRGAN leads NAF-DPM by >= 2 pages
    - NAF-DPM packs patches from multiple pages to fill batch_size * 2
    - Fixed batch shape (pad, don't vary) to avoid DML recompilation

Progress: 5 parallel tqdm bars (one per stage)
"""

import sys
import os
import gc
import time
import threading
import queue
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Callable
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import onnxruntime as ort
import doxapy
import fitz  # PyMuPDF
from tqdm import tqdm

# 限制 OpenCV 内部线程，避免与流水线线程池过度订阅。
cv2.setNumThreads(1)

# Add script directory for local imports
if getattr(sys, 'frozen', False):
    # PyInstaller打包后的临时解压目录
    SCRIPT_DIR = Path(sys._MEIPASS)
else:
    SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from nafdpm.schedule.schedule import Schedule
from nafdpm.schedule.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver

# Reuse GPU detection and session options from ml_enhance
from ml_enhance import (
    get_gpu_providers,
    create_session_options,
    OnnxInitPredictor,
    OnnxDenoiser,
    nafdpm_crop_concat,
    nafdpm_crop_concat_back,
    create_dpm_solver_context,
    dynamic_thresholding,
    create_protection_mask,
)

logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

# SLBR
SLBR_TILE_SIZE = 256
SLBR_OVERLAP = 64
SLBR_THRESHOLD = 0.3
SLBR_MIN_AREA = 2000
SLBR_INITIAL_CPU_WORKERS = 3
SLBR_MAX_CPU_WORKERS_RATIO = 0.75  # max 75% of CPU cores

# ESRGAN
ESRGAN_TILE_SIZE = 128
ESRGAN_DEFAULT_OVERLAP = 8
ESRGAN_SKIP_NO_WM = os.getenv("FW_ESRGAN_SKIP_NO_WM", "0") == "1"

# NAF-DPM
NAFDPM_DEFAULT_BATCH_SIZE = 32
NAFDPM_DPM_STEPS = 20
NAFDPM_PATCH_SIZE = 128

# DoxaPy
DOXA_DEFAULT_BLEND = 0.20
DOXA_CPU_WORKERS = 2
DOXA_STRETCH_MIN_RANGE = 20

# Pipeline
LEAD_PAGES = 2       # min pages each stage should lead the next
QUEUE_MAXSIZE = 8     # bounded queue size between stages
SLBR_MAX_LEAD = 5     # SLBR max pages ahead of ESRGAN
RENDER_MAX_LEAD = 5   # Render max pages ahead of SLBR
NAFDPM_PAGE_AWARE_PACKING = os.getenv("FW_PAGE_AWARE_PACKING", "1") == "1"
NAFDPM_ENQUEUE_BURST = 0
NAFDPM_BATCH_QUOTA_WHEN_ESRGAN_BACKLOG = int(os.getenv("FW_NAFDPM_BATCH_QUOTA", "2"))
ESRGAN_PREP_OFFLOAD = os.getenv("FW_ESRGAN_PREP_OFFLOAD", "1") == "1"
NAFDPM_INPUT_LAYOUT_CACHE = os.getenv("FW_NAFDPM_INPUT_LAYOUT_CACHE", "0") == "1"


# ============================================================
# Data Structures
# ============================================================

class Stage(Enum):
    """Pipeline stage identifiers."""
    RENDER = auto()
    SLBR = auto()
    ESRGAN = auto()
    NAFDPM = auto()
    DOXA = auto()


@dataclass
class PageData:
    """Data passed between pipeline stages for a single page.

    Fields are populated/consumed by different stages:
      - page_idx, image_bgr: set by PageLoader
      - watermark_boxes: set by SLBR
      - esrgan_result: set by ESRGAN (float32 RGB [0,1] with QR mask applied)
      - nafdpm_result: set by NAF-DPM (uint8 BGR)
      - final_gray: set by DoxaPy (uint8 grayscale)
      - error: set by any stage on failure
    """
    page_idx: int
    image_bgr: np.ndarray                          # input BGR uint8 from page render

    # SLBR output
    watermark_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)

    # ESRGAN output -- float32 RGB [0,1] for direct NAF-DPM consumption
    esrgan_result: Optional[np.ndarray] = None
    # Cached NAF-DPM input layout (1,3,H,W) float32 derived from esrgan_result
    nafdpm_input_chw: Optional[np.ndarray] = None
    # Optional precomputed ESRGAN tile inputs (CPU-offloaded)
    esrgan_prepared: Optional[Any] = None

    # NAF-DPM output -- uint8 BGR
    nafdpm_result: Optional[np.ndarray] = None

    # DoxaPy output -- uint8 grayscale
    final_gray: Optional[np.ndarray] = None

    # Error tracking
    error: Optional[str] = None
    error_stage: Optional[Stage] = None


@dataclass
class PatchInfo:
    """Tracks a single 128x128 patch across cross-page NAF-DPM batch packing.

    When NAF-DPM packs patches from multiple pages into one batch,
    we need to know which page each patch belongs to for reassembly.
    """
    page_idx: int
    patch_local_idx: int   # index within this page's patch array
    is_padding: bool = False  # True if this is a dummy patch to fill batch


@dataclass
class NAFDPMPageAccumulator:
    """Collects NAF-DPM results for a page until all patches are processed.

    A page may have its patches split across multiple NAF-DPM batches
    (cross-page packing), so we accumulate results here.
    """
    page_idx: int
    total_patches: int
    grid_info: Tuple[int, int, int, int]  # (n_h, n_w, orig_h, orig_w)
    init_predicts: np.ndarray             # (N, 3, 128, 128) -- init predictor output
    result_patches: Optional[np.ndarray] = None  # (N, 3, 128, 128) -- will be filled
    patches_completed: int = 0

    def is_complete(self) -> bool:
        return self.patches_completed >= self.total_patches


# ============================================================
# Sentinel
# ============================================================

_SENTINEL = object()  # signals end-of-stream in queues


# ============================================================
# SLBR Worker
# ============================================================

class SLBRWorker:
    """Watermark detection via SLBR model.

    Primary mode: CPU thread pool (each thread has its own ORT CPU session).
    Fallback mode: GPU on main thread (when CPU can't keep pace).
    """

    def __init__(self, model_path: Path, num_cpu_workers: int = SLBR_INITIAL_CPU_WORKERS):
        self.model_path = model_path
        self.num_cpu_workers = num_cpu_workers
        self.use_gpu_fallback = False  # set True dynamically if CPU too slow

        # CPU sessions (one per worker thread, created lazily in thread)
        self._thread_local = threading.local()

        # GPU session (shared, only used on main thread)
        self._gpu_session: Optional[ort.InferenceSession] = None

        # Stats for dynamic scaling
        self.pages_completed = 0
        self.total_cpu_time = 0.0

    def _get_cpu_session(self) -> ort.InferenceSession:
        """Get or create a per-thread ORT CPU session.

        Each CPU worker gets its own session with limited thread usage:
          intra_op_num_threads=2, inter_op_num_threads=1
        This prevents CPU contention across workers.
        """
        session = getattr(self._thread_local, 'session', None)
        if session is None:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.log_severity_level = 3  # Only show errors
            opts.intra_op_num_threads = 2
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session = ort.InferenceSession(
                str(self.model_path), opts,
                providers=['CPUExecutionProvider']
            )
            self._thread_local.session = session
        return session

    def _get_gpu_session(self) -> ort.InferenceSession:
        """Get or create the GPU session (main thread only)."""
        if self._gpu_session is None:
            providers, _ = get_gpu_providers()
            opts = create_session_options()
            self._gpu_session = ort.InferenceSession(
                str(self.model_path), opts,
                providers=providers
            )
        return self._gpu_session

    def _detect_watermark_impl(self, image_bgr: np.ndarray,
                                session: ort.InferenceSession) -> List[Tuple[int, int, int, int]]:
        """Core SLBR inference logic (shared by CPU and GPU paths).

        Tiled inference with weighted averaging for overlapping regions.
        Returns list of (x_min, y_min, x_max, y_max) watermark bounding boxes.
        """
        h, w = image_bgr.shape[:2]
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        input_name = session.get_inputs()[0].name

        mask_sum = np.zeros((h, w), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)
        stride = SLBR_TILE_SIZE - SLBR_OVERLAP

        # Pre-calculate tile coordinates
        tiles = []
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y_end = min(y + SLBR_TILE_SIZE, h)
                x_end = min(x + SLBR_TILE_SIZE, w)
                y_start = max(0, y_end - SLBR_TILE_SIZE)
                x_start = max(0, x_end - SLBR_TILE_SIZE)
                tiles.append((y_start, y_end, x_start, x_end))

        for y_start, y_end, x_start, x_end in tiles:
            tile = img_rgb[y_start:y_end, x_start:x_end]

            pad_h = SLBR_TILE_SIZE - tile.shape[0]
            pad_w = SLBR_TILE_SIZE - tile.shape[1]
            if pad_h > 0 or pad_w > 0:
                tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

            tile_input = tile.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 256, 256)
            mask_tile = session.run(None, {input_name: tile_input})[0][0, 0]  # (256, 256)

            if pad_h > 0:
                mask_tile = mask_tile[:-pad_h]
            if pad_w > 0:
                mask_tile = mask_tile[:, :-pad_w]

            mask_sum[y_start:y_end, x_start:x_end] += mask_tile
            weight_sum[y_start:y_end, x_start:x_end] += 1

        mask_avg = mask_sum / np.maximum(weight_sum, 1e-6)

        # Extract bounding boxes from mask
        mask_uint8 = (mask_avg * 255).clip(0, 255).astype(np.uint8)
        _, binary = cv2.threshold(mask_uint8, int(SLBR_THRESHOLD * 255), 255, cv2.THRESH_BINARY)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        watermark_boxes = []
        margin = 30
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch
            aspect_ratio = float(cw) / ch if ch > 0 else 0

            if area > SLBR_MIN_AREA and 0.3 < aspect_ratio < 3.0:
                x_min = max(0, x - margin)
                y_min = max(0, y - margin)
                x_max = min(w, x + cw + margin)
                y_max = min(h, y + ch + margin)
                watermark_boxes.append((x_min, y_min, x_max, y_max))

        return watermark_boxes

    def detect_watermark_cpu(self, page: PageData) -> PageData:
        """Run SLBR watermark detection on CPU (called from thread pool)."""
        t0 = time.perf_counter()
        try:
            session = self._get_cpu_session()
            page.watermark_boxes = self._detect_watermark_impl(page.image_bgr, session)
        except Exception as e:
            logger.warning(f"SLBR CPU failed for page {page.page_idx}: {e}")
            page.watermark_boxes = []
        elapsed = time.perf_counter() - t0
        self.total_cpu_time += elapsed
        self.pages_completed += 1
        return page

    def detect_watermark_gpu(self, page: PageData) -> PageData:
        """Run SLBR watermark detection on GPU (called from main thread)."""
        try:
            session = self._get_gpu_session()
            page.watermark_boxes = self._detect_watermark_impl(page.image_bgr, session)
        except Exception as e:
            logger.warning(f"SLBR GPU failed for page {page.page_idx}: {e}")
            page.watermark_boxes = []
        self.pages_completed += 1
        return page


# ============================================================
# ESRGAN Worker
# ============================================================

class ESRGANWorker:
    """ESRGAN tiled super-resolution on GPU.

    Uses IO binding for DirectML to minimize allocation overhead.
    Output is float32 RGB [0,1] for direct handoff to NAF-DPM.
    QR protection mask is applied here.
    """

    def __init__(self, model_path: Path, tile_size: int = ESRGAN_TILE_SIZE,
                 overlap: int = ESRGAN_DEFAULT_OVERLAP):
        self.model_path = model_path
        self.tile_size = tile_size
        self.overlap = overlap

        self._session: Optional[ort.InferenceSession] = None
        self._io_binding = None
        self._input_buffer: Optional[np.ndarray] = None
        self._output_buffer: Optional[np.ndarray] = None
        self._input_name: Optional[str] = None
        self._use_dml: bool = False

        # Stats
        self.pages_completed = 0

    def _ensure_session(self):
        """Load ESRGAN session (lazy, main thread)."""
        if self._session is not None:
            return

        providers, provider_name = get_gpu_providers()
        opts = create_session_options()
        self._session = ort.InferenceSession(
            str(self.model_path), opts, providers=providers
        )
        self._input_name = self._session.get_inputs()[0].name

    def _run_tile(self, tile_chw: np.ndarray) -> np.ndarray:
        """Run inference on a single tile (3, 128, 128) -> (3, H, W).

        Uses direct session.run for both CPU and DML.
        IO binding has a bug that returns garbage on first call.
        """
        batch = tile_chw[np.newaxis, ...]  # (1, 3, 128, 128)
        result = self._session.run(None, {self._input_name: batch})[0]
        return result[0]

    def prepare_tiles(self, image_bgr: np.ndarray):
        """CPU预处理：预先生成 ESRGAN 需要的 tile 输入。

        返回: (h, w, tiles_info, tile_inputs, pad_infos, original_rgb)
        """
        image = image_bgr
        h, w = image.shape[:2]
        stride = self.tile_size - self.overlap

        original_rgb = None

        tiles_info = []
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y_end = min(y + self.tile_size, h)
                x_end = min(x + self.tile_size, w)
                y_start = max(0, y_end - self.tile_size)
                x_start = max(0, x_end - self.tile_size)
                tiles_info.append((y_start, y_end, x_start, x_end))

        tile_inputs = []
        pad_infos = []
        for y_start, y_end, x_start, x_end in tiles_info:
            tile = image[y_start:y_end, x_start:x_end]
            pad_h = self.tile_size - tile.shape[0]
            pad_w = self.tile_size - tile.shape[1]
            if pad_h > 0 or pad_w > 0:
                tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            pad_infos.append((pad_h, pad_w))
            tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tile_inputs.append(np.transpose(tile_rgb, (2, 0, 1)))

        return h, w, tiles_info, tile_inputs, pad_infos, original_rgb

    def process_page(self, page: PageData, progress_callback=None, prepared=None) -> PageData:
        """Run ESRGAN on a page, apply QR mask, store float32 RGB result.

        Input: page.image_bgr (uint8 BGR)
        Output: page.esrgan_result (float32 RGB [0,1])
        """
        self._ensure_session()
        image = page.image_bgr

        # Algorithmic fast path:
        # For pages with no SLBR-detected watermark regions, optionally skip
        # full-page ESRGAN tiling to reduce dominant GPU workload.
        if ESRGAN_SKIP_NO_WM and not page.watermark_boxes:
            page.esrgan_result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            self.pages_completed += 1
            if progress_callback:
                progress_callback(1, 1)
            return page

        original_rgb = None
        if prepared is None:
            h, w, tiles_info, tile_inputs, pad_infos, original_rgb = self.prepare_tiles(image)
        else:
            # Backward compatibility with old prepared payload shape.
            if len(prepared) == 6:
                h, w, tiles_info, tile_inputs, pad_infos, original_rgb = prepared
            else:
                h, w, tiles_info, tile_inputs, pad_infos = prepared

        total_tiles = len(tiles_info)

        # Phase 2: Inference + accumulation
        output = np.zeros((h, w, 3), dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)

        for idx in range(total_tiles):
            raw_result = self._run_tile(tile_inputs[idx])  # (3, H_out, W_out)

            # Post-process tile
            result_tile = np.transpose(raw_result, (1, 2, 0))  # (H, W, 3) RGB float32
            result_tile = np.clip(result_tile, 0, 1)
            if result_tile.shape[0] != self.tile_size:
                result_tile = cv2.resize(result_tile, (self.tile_size, self.tile_size),
                                         interpolation=cv2.INTER_AREA)

            # Remove padding
            pad_h, pad_w = pad_infos[idx]
            if pad_h > 0:
                result_tile = result_tile[:-pad_h]
            if pad_w > 0:
                result_tile = result_tile[:, :-pad_w]

            y_start, y_end, x_start, x_end = tiles_info[idx]
            output[y_start:y_end, x_start:x_end] += result_tile
            weight[y_start:y_end, x_start:x_end] += 1

            if progress_callback:
                progress_callback(idx + 1, total_tiles)

        output = output / np.maximum(weight[:, :, np.newaxis], 1)

        # Apply QR/watermark protection mask
        if page.watermark_boxes:
            protection_mask = create_protection_mask(image.shape, page.watermark_boxes, feather=30)
            mask_3ch = protection_mask[:, :, np.newaxis]
            if original_rgb is None:
                original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            output = output * mask_3ch + original_rgb * (1 - mask_3ch)

        page.esrgan_result = output  # float32 RGB [0,1]
        if NAFDPM_INPUT_LAYOUT_CACHE:
            page.nafdpm_input_chw = np.transpose(output, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
        else:
            page.nafdpm_input_chw = None
        self.pages_completed += 1

        if progress_callback:
            progress_callback(total_tiles, total_tiles)

        return page


# ============================================================
# NAF-DPM Worker
# ============================================================

class NAFDPMWorker:
    """NAF-DPM denoising on GPU with cross-page batch packing.

    Key features:
    - Packs patches from multiple pages to fill batch_size * 2
    - Uses PatchInfo to track which page each patch belongs to
    - Fixed batch shape (pads with dummy patches) to avoid DML recompilation
    - Per-page accumulator for reassembly
    """

    def __init__(self, models_dir: Path, batch_size: int = NAFDPM_DEFAULT_BATCH_SIZE,
                 dpm_steps: int = NAFDPM_DPM_STEPS):
        self.models_dir = models_dir
        self.batch_size = batch_size
        self.dpm_steps = dpm_steps
        # Effective batch = 2 * batch_size (process two batches in parallel)
        self.effective_batch = batch_size * 2

        # Models (lazy loaded on main thread)
        self._init_predictor: Optional[OnnxInitPredictor] = None
        self._denoiser: Optional[OnnxDenoiser] = None
        self._schedule: Optional[Schedule] = None
        self._noise_schedule = None

        # Cross-page batch packing state
        # Pending patches pool: list of (patch_array(3,128,128), PatchInfo)
        self._pending_patches: List[Tuple[np.ndarray, PatchInfo]] = []
        # Per-page accumulators: page_idx -> NAFDPMPageAccumulator
        self._accumulators: Dict[int, NAFDPMPageAccumulator] = {}

        # RNG for reproducible noise
        self._rng = np.random.default_rng(42)

        # Stats
        self.pages_completed = 0
        self.total_patches_processed = 0

    def _ensure_models(self):
        """Load NAF-DPM models (lazy, main thread)."""
        if self._init_predictor is not None:
            return

        from ml_enhance import load_nafdpm_onnx
        self._init_predictor, self._denoiser, self._schedule = \
            load_nafdpm_onnx(self.models_dir)

        # Pre-build noise schedule
        self._noise_schedule, _ = create_dpm_solver_context(
            self._schedule, self._denoiser)

    def enqueue_page(self, page: PageData):
        """Add a page's patches to the pending pool for batch packing.

        Extracts patches from esrgan_result, runs init_predictor on them,
        creates accumulator, and adds patches to the pending pool.
        """
        self._ensure_models()

        if page.nafdpm_input_chw is not None:
            img_array = page.nafdpm_input_chw
        else:
            image_rgb = page.esrgan_result  # float32 RGB [0,1]
            img_array = np.transpose(image_rgb, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        patches, grid_info = nafdpm_crop_concat(img_array, size=NAFDPM_PATCH_SIZE)
        num_patches = patches.shape[0]

        # Run init_predictor for all patches (GPU, on main thread)
        # Process in sub-batches to avoid OOM on large pages
        init_preds_list = []
        for b_start in range(0, num_patches, self.batch_size):
            b_end = min(b_start + self.batch_size, num_patches)
            batch = patches[b_start:b_end]
            init_pred = self._init_predictor(batch)
            init_preds_list.append(init_pred)
        init_predicts = np.concatenate(init_preds_list, axis=0)

        # Create accumulator
        acc = NAFDPMPageAccumulator(
            page_idx=page.page_idx,
            total_patches=num_patches,
            grid_info=grid_info,
            init_predicts=init_predicts,
            result_patches=np.zeros_like(init_predicts),
        )
        self._accumulators[page.page_idx] = acc

        # Add patches to pending pool with tracking info
        for i in range(num_patches):
            self._pending_patches.append((
                patches[i],  # (3, 128, 128) original patch
                PatchInfo(page_idx=page.page_idx, patch_local_idx=i)
            ))

        # Free esrgan_result to save memory (patches are extracted)
        page.esrgan_result = None
        page.nafdpm_input_chw = None

    def pending_patch_count(self) -> int:
        """Number of patches waiting to be processed."""
        return len(self._pending_patches)

    def has_work(self) -> bool:
        """Check if there are enough patches to form a batch.

        Returns True if we have >= effective_batch patches,
        OR if we have any patches and no more pages are coming (flush mode).
        """
        return len(self._pending_patches) > 0

    def has_full_batch(self) -> bool:
        """Check if we have enough patches for a full effective batch."""
        return len(self._pending_patches) >= self.effective_batch

    def process_batch(self, progress_callback=None) -> List[PageData]:
        """Process one batch of patches (possibly from multiple pages).

        Takes up to effective_batch patches from the pending pool.
        Pads with dummy patches if needed to maintain fixed batch shape.
        Runs the full DPM solver loop (dpm_steps iterations).

        Returns list of PageData for pages that are now fully complete.
        """
        self._ensure_models()

        if not self._pending_patches:
            return []

        # Take patches for this batch
        take_count = min(len(self._pending_patches), self.effective_batch)
        if NAFDPM_PAGE_AWARE_PACKING and len(self._pending_patches) > take_count:
            buckets: Dict[int, List[Tuple[np.ndarray, PatchInfo]]] = {}
            page_order: List[int] = []
            for item in self._pending_patches:
                pid = item[1].page_idx
                if pid not in buckets:
                    buckets[pid] = []
                    page_order.append(pid)
                buckets[pid].append(item)

            ranked_pages = sorted(page_order, key=lambda p: len(buckets[p]), reverse=True)
            batch_patches_info: List[Tuple[np.ndarray, PatchInfo]] = []
            remain = take_count
            for pid in ranked_pages:
                if remain <= 0:
                    break
                bucket = buckets[pid]
                take_n = min(len(bucket), remain)
                if take_n > 0:
                    batch_patches_info.extend(bucket[:take_n])
                    buckets[pid] = bucket[take_n:]
                    remain -= take_n

            new_pending: List[Tuple[np.ndarray, PatchInfo]] = []
            for pid in page_order:
                new_pending.extend(buckets[pid])
            self._pending_patches = new_pending
        else:
            batch_patches_info = self._pending_patches[:take_count]
            self._pending_patches = self._pending_patches[take_count:]

        # Separate data and tracking info
        real_patches = []
        real_infos = []
        for patch_arr, info in batch_patches_info:
            real_patches.append(patch_arr)
            real_infos.append(info)

        real_count = len(real_patches)

        # Pad to fixed effective_batch size if needed (avoid DML recompilation)
        pad_count = self.effective_batch - real_count
        if pad_count > 0:
            dummy = np.zeros((3, NAFDPM_PATCH_SIZE, NAFDPM_PATCH_SIZE), dtype=np.float32)
            for _ in range(pad_count):
                real_patches.append(dummy)
                real_infos.append(PatchInfo(page_idx=-1, patch_local_idx=-1, is_padding=True))

        patches_array = np.stack(real_patches, axis=0)  # (EB, 3, 128, 128)

        # Gather init_predicts for the full effective batch
        init_preds_all = np.zeros_like(patches_array)
        for i, info in enumerate(real_infos):
            if not info.is_padding:
                acc = self._accumulators[info.page_idx]
                init_preds_all[i] = acc.init_predicts[info.patch_local_idx]

        # ---- Split into 2 sub-batches for step-interleaved DPM ----
        # Each denoiser GPU call processes batch_size (not effective_batch) patches.
        # Two normal batch_size batches naturally combine into effective_batch.
        # Step interleaving: batchA_step0 -> batchB_step0 -> batchA_step1 -> ...
        # While batchB runs on GPU, batchA's CPU work from same step completes.
        num_sub = 2
        sub_init_preds = [
            init_preds_all[b * self.batch_size:(b + 1) * self.batch_size]
            for b in range(num_sub)
        ]

        # Generate independent initial noise per sub-batch
        x_states = [
            self._rng.standard_normal(
                (self.batch_size, 3, NAFDPM_PATCH_SIZE, NAFDPM_PATCH_SIZE)
            ).astype(np.float32)
            for _ in range(num_sub)
        ]

        # Pre-compute timesteps (singlestep, order=1, time_uniform)
        ns = self._noise_schedule
        total_N = ns.total_N  # 100
        t_T = ns.T
        t_0 = 1. / total_N
        timesteps = np.linspace(t_T, t_0, self.dpm_steps + 1).astype(np.float32)

        # Step-interleaved DPM solver loop (double-layer)
        # Outer loop: DPM steps  |  Inner loop: sub-batches (A, B)
        # Correctness guarantee: when batchA is next used (step S+1),
        # its step S CPU work was already joined during batchB's processing.
        cpu_thread = None
        cpu_result = [None, None]  # [batch_idx, x_t]

        for step in range(self.dpm_steps):
            s_val = timesteps[step]
            t_val = timesteps[step + 1]
            lambda_s = ns.marginal_lambda(s_val)
            lambda_t = ns.marginal_lambda(t_val)
            h = lambda_t - lambda_s
            log_alpha_t = ns.marginal_log_mean_coeff(t_val)
            sigma_s = ns.marginal_std(s_val)
            sigma_t = ns.marginal_std(t_val)
            alpha_t = np.exp(log_alpha_t)
            phi_1 = np.expm1(-h)
            t_input = float((s_val - 1. / total_N) * 1000.)
            t_arr = np.full((self.batch_size,), np.float32(t_input), dtype=np.float32)

            for b in range(num_sub):
                x = x_states[b]
                cond = sub_init_preds[b]

                # === GPU: denoiser (main thread, batch_size patches) ===
                x0_pred = self._denoiser(x, t_arr, cond)

                # Join PREVIOUS sub-batch's CPU work (different sub-batch)
                if cpu_thread is not None:
                    cpu_thread.join()
                    x_states[cpu_result[0]] = cpu_result[1]

                # === CPU: thresholding + solver arithmetic (worker) ===
                def _cpu_step(_b, _x, _x0, _ss, _st, _at, _p1):
                    x0_c = dynamic_thresholding(_x0)
                    cpu_result[0] = _b
                    cpu_result[1] = (_st / _ss) * _x - _at * _p1 * x0_c

                cpu_thread = threading.Thread(
                    target=_cpu_step,
                    args=(b, x, x0_pred, sigma_s, sigma_t, alpha_t, phi_1))
                cpu_thread.start()

            if progress_callback:
                progress_callback(step + 1, self.dpm_steps)

        # Join final CPU thread
        if cpu_thread is not None:
            cpu_thread.join()
            x_states[cpu_result[0]] = cpu_result[1]

        # Combine sub-batch results and add init_predicts
        result = np.concatenate([
            np.clip(x_states[b] + sub_init_preds[b], 0, 1)
            for b in range(num_sub)
        ], axis=0)

        # Distribute results back to page accumulators
        completed_pages = []
        for i, info in enumerate(real_infos):
            if info.is_padding:
                continue
            acc = self._accumulators[info.page_idx]
            acc.result_patches[info.patch_local_idx] = result[i]
            acc.patches_completed += 1

        self.total_patches_processed += real_count

        # Check which pages are complete (deduplicate page_idxs to avoid
        # accessing an accumulator that was already deleted in this loop)
        seen_page_idxs = set()
        for info in real_infos:
            if info.is_padding or info.page_idx in seen_page_idxs:
                continue
            seen_page_idxs.add(info.page_idx)
            acc = self._accumulators.get(info.page_idx)
            if acc is None:
                continue
            if acc.is_complete():
                # Reconstruct full image from patches
                reconstructed = nafdpm_crop_concat_back(
                    acc.result_patches, acc.grid_info, size=NAFDPM_PATCH_SIZE)
                result_np = np.transpose(reconstructed[0], (1, 2, 0))
                result_np = np.clip(result_np * 255, 0, 255).astype(np.uint8)
                result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

                page = PageData(page_idx=acc.page_idx, image_bgr=np.empty(0, dtype=np.uint8))
                page.nafdpm_result = result_bgr
                completed_pages.append(page)

                # Clean up accumulator
                del self._accumulators[acc.page_idx]
                self.pages_completed += 1

        return completed_pages


# ============================================================
# DoxaPy Worker
# ============================================================

class DoxaPyWorker:
    """DoxaPy Sauvola binarization + blending on CPU thread pool."""

    def __init__(self, blend_ratio: float = DOXA_DEFAULT_BLEND,
                 num_workers: int = DOXA_CPU_WORKERS):
        self.blend_ratio = blend_ratio
        self.num_workers = num_workers
        self.pages_completed = 0

    def process_page(self, page: PageData) -> PageData:
        """Apply DoxaPy binarization and blending, produce final grayscale.

        Input: page.nafdpm_result (uint8 BGR)
        Output: page.final_gray (uint8 grayscale)
        """
        try:
            image_bgr = page.nafdpm_result
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            gray = np.ascontiguousarray(gray, dtype=np.uint8)
            binary = np.zeros_like(gray, dtype=np.uint8)
            binarizer = doxapy.Binarization(doxapy.Binarization.Algorithms.SAUVOLA)
            binarizer.initialize(gray)
            binarizer.to_binary(binary)
            blended = cv2.addWeighted(gray, 1 - self.blend_ratio, binary, self.blend_ratio, 0)
            
            # === 灰度拉伸: 百分位裁剪 + 线性映射到 0-255 ===
            # 保持相对灰度强度不变，只是线性拉伸范围
            # 这样文字会更黑、背景会更白，同时不破坏半透明水印
            low_p = np.percentile(blended, 1)
            high_p = np.percentile(blended, 99)
            # 动态范围门槛：文字稀疏页通常 high_p-low_p 很小，
            # 强行拉伸会导致整体发黑。
            if high_p > low_p and (high_p - low_p) >= DOXA_STRETCH_MIN_RANGE:
                blended = np.clip(blended, low_p, high_p)
                blended = ((blended - low_p) / (high_p - low_p) * 255).astype(np.uint8)
            
            page.final_gray = blended
        except Exception as e:
            logger.warning(f"DoxaPy failed for page {page.page_idx}: {e}")
            # Fallback: simple grayscale conversion
            page.final_gray = cv2.cvtColor(page.nafdpm_result, cv2.COLOR_BGR2GRAY)
            page.error = f"DoxaPy: {e}"
            page.error_stage = Stage.DOXA
        self.pages_completed += 1
        return page


# ============================================================
# Progress Manager
# ============================================================

class ProgressManager:
    """Manages 5 parallel tqdm progress bars for the pipeline stages.

    Each bar tracks one stage's completion across all pages.
    Thread-safe: uses a lock for updates from background threads.

    Bar layout (top to bottom):
      position=0: SLBR    watermark detection
      position=1: ESRGAN  super-resolution
      position=2: NAF-DPM denoising
      position=3: DoxaPy  binarization
    """

    STAGE_NAMES = {
        Stage.RENDER: "Render ",
        Stage.SLBR: "SLBR   ",
        Stage.ESRGAN: "ESRGAN ",
        Stage.NAFDPM: "NAF-DPM",
        Stage.DOXA: "DoxaPy ",
    }
    STAGE_POSITIONS = {
        Stage.RENDER: 0,
        Stage.SLBR: 1,
        Stage.ESRGAN: 2,
        Stage.NAFDPM: 3,
        Stage.DOXA: 4,
    }

    BAR_FORMAT = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
    STATUS_REFRESH_INTERVAL = 0.2

    def __init__(self, total_pages: int):
        self.total_pages = total_pages
        self._lock = threading.Lock()
        self._bars: Dict[Stage, tqdm] = {}
        self._stage_status: Dict[Stage, str] = {}
        self._last_status_refresh: Dict[Stage, float] = {}

    def create_bars(self):
        """Create the 4 tqdm progress bars."""
        for stage in [Stage.RENDER, Stage.SLBR, Stage.ESRGAN, Stage.NAFDPM, Stage.DOXA]:
            name = self.STAGE_NAMES[stage]
            pos = self.STAGE_POSITIONS[stage]
            bar = tqdm(
                total=self.total_pages,
                desc=f"      {name}",
                position=pos,
                leave=True,
                bar_format=self.BAR_FORMAT,
            )
            self._bars[stage] = bar
            self._stage_status[stage] = ""
            self._last_status_refresh[stage] = 0.0

    def update(self, stage: Stage, increment: int = 1, status: str = ""):
        """Thread-safe progress update."""
        with self._lock:
            bar = self._bars.get(stage)
            if bar is not None:
                bar.update(increment)
                if status:
                    now = time.perf_counter()
                    if now - self._last_status_refresh.get(stage, 0.0) >= self.STATUS_REFRESH_INTERVAL:
                        bar.set_postfix_str(status, refresh=True)
                        self._last_status_refresh[stage] = now

    def set_status(self, stage: Stage, status: str):
        """Update the postfix/status text for a stage bar."""
        with self._lock:
            bar = self._bars.get(stage)
            if bar is not None:
                now = time.perf_counter()
                if now - self._last_status_refresh.get(stage, 0.0) >= self.STATUS_REFRESH_INTERVAL:
                    bar.set_postfix_str(status, refresh=True)
                    self._last_status_refresh[stage] = now

    def close(self):
        """Close all progress bars."""
        with self._lock:
            for bar in self._bars.values():
                bar.close()
            self._bars.clear()


# ============================================================
# Main Pipeline Processor
# ============================================================

class PipelinedMLProcessor:
    """5-stage producer-consumer ML document enhancement pipeline.

    Usage:
        processor = PipelinedMLProcessor(models_dir)
        results = processor.process_document(pdf_path, page_indices, dpi=150)
        # results: dict of {page_idx: grayscale_uint8_array}

    All GPU work is serialized on the calling thread (main thread).
    CPU work (SLBR, DoxaPy) runs in background thread pools.
    PDF rendering runs in a background thread with bounded queue.
    """

    def __init__(self, models_dir: Optional[Path] = None,
                 nafdpm_batch_size: int = NAFDPM_DEFAULT_BATCH_SIZE,
                 esrgan_overlap: int = ESRGAN_DEFAULT_OVERLAP,
                 doxa_blend: float = DOXA_DEFAULT_BLEND,
                 dpm_steps: int = NAFDPM_DPM_STEPS):
        if models_dir is None:
            models_dir = SCRIPT_DIR / "models"
        self.models_dir = Path(models_dir)

        # Pipeline config
        self.nafdpm_batch_size = nafdpm_batch_size
        self.esrgan_overlap = esrgan_overlap
        self.doxa_blend = doxa_blend
        self.dpm_steps = dpm_steps

        # Workers (created during process_pages)
        self._slbr: Optional[SLBRWorker] = None
        self._esrgan: Optional[ESRGANWorker] = None
        self._nafdpm: Optional[NAFDPMWorker] = None
        self._doxa: Optional[DoxaPyWorker] = None

        # Inter-stage queues (bounded)
        self._q_render_out: Optional[queue.Queue] = None    # Render -> SLBR
        self._q_slbr_out: Optional[queue.Queue] = None      # SLBR -> ESRGAN (main thread)
        self._q_slbr_gpu: Optional[queue.Queue] = None       # SLBR CPU-failed -> GPU fallback
        self._q_nafdpm_out: Optional[queue.Queue] = None     # NAF-DPM -> DoxaPy
        self._q_doxa_out: Optional[queue.Queue] = None       # DoxaPy -> results

        # Progress
        self._progress: Optional[ProgressManager] = None


    def _create_workers(self):
        """Initialize all worker instances."""
        self._slbr = SLBRWorker(
            model_path=self.models_dir / "slbr-watermark-detector-fp16.onnx",
            num_cpu_workers=SLBR_INITIAL_CPU_WORKERS,
        )
        self._esrgan = ESRGANWorker(
            model_path=self.models_dir / "real-esrgan-x4plus-128-fp16.onnx",
            overlap=self.esrgan_overlap,
        )
        self._nafdpm = NAFDPMWorker(
            models_dir=self.models_dir,
            batch_size=self.nafdpm_batch_size,
            dpm_steps=self.dpm_steps,
        )
        self._doxa = DoxaPyWorker(
            blend_ratio=self.doxa_blend,
        )

    def _create_queues(self):
        """Initialize inter-stage queues.

        _q_doxa_out is unbounded to prevent deadlock:
        Main thread reads _q_doxa_out only AFTER _gpu_main_loop finishes,
        so a bounded queue would block DoxaPy -> block _q_nafdpm_out consumer
        -> block main thread _q_nafdpm_out producer -> deadlock.
        """
        # Render queue: just a handoff channel (maxsize=1).
        # Total render lead = _q_render_out.qsize() + pending_futures count
        # The pending_futures cap = RENDER_MAX_LEAD ensures total <= RENDER_MAX_LEAD.
        self._q_render_out = queue.Queue(maxsize=1)
        # SLBR output queue: reduced so that _q_slbr_out + esrgan_queue <= SLBR_MAX_LEAD.
        # esrgan_queue is capped at LEAD_PAGES, so maxsize = SLBR_MAX_LEAD - LEAD_PAGES.
        self._q_slbr_out = queue.Queue(maxsize=max(1, SLBR_MAX_LEAD - LEAD_PAGES))
        self._q_slbr_gpu = queue.Queue()  # CPU-failed pages for GPU fallback
        self._q_nafdpm_out = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._q_doxa_out = queue.Queue()  # unbounded — prevents deadlock

    def _render_loop(self, pdf_path: str, page_indices: List[int], dpi: int):
        """Background thread: render PDF pages to BGR images.

        Puts PageData into _q_render_out (bounded at RENDER_MAX_LEAD=5),
        so this naturally blocks when 5 pages ahead of SLBR consumer.
        """
        doc = fitz.open(pdf_path)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        for page_num in page_indices:
            if page_num >= len(doc):
                logger.warning(f"Render: page {page_num} out of range, skipping")
                continue
            pix = doc[page_num].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3).copy()  # copy: pix buffer is transient
            image_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            page_data = PageData(page_idx=page_num, image_bgr=image_bgr)
            # Bounded put -> blocks when RENDER_MAX_LEAD ahead of SLBR
            self._q_render_out.put(page_data)
            if self._progress:
                self._progress.update(Stage.RENDER, status=f"p{page_num+1}")

        doc.close()
        self._q_render_out.put(_SENTINEL)

    def _slbr_cpu_loop(self):
        """Background thread: read pages from render queue, run SLBR on CPU,
        feed results to _q_slbr_out.

        Lead limiting:
          - Input (_q_render_out) is bounded at RENDER_MAX_LEAD -> render pauses
          - Output (_q_slbr_out) is bounded at SLBR_MAX_LEAD -> SLBR pauses
        This naturally throttles SLBR to be at most SLBR_MAX_LEAD pages
        ahead of ESRGAN (who consumes _q_slbr_out).

        If a page fails on CPU, it's routed to _q_slbr_gpu for GPU fallback
        on the main thread.
        """
        max_workers = max(1, int(os.cpu_count() * SLBR_MAX_CPU_WORKERS_RATIO))
        num_workers = min(self._slbr.num_cpu_workers, max_workers)

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            pending_futures = []  # ordered: [(future, original_page), ...]
            render_done = False

            while True:
                did_work = False

                # 1. Submit new pages from render queue (non-blocking)
                # Total render lead = _q_render_out.qsize() + len(pending_futures)
                # _q_render_out maxsize=1, so cap pending at RENDER_MAX_LEAD-1
                # to ensure total <= RENDER_MAX_LEAD (1 + 4 = 5).
                if not render_done and len(pending_futures) < RENDER_MAX_LEAD - 1:
                    try:
                        item = self._q_render_out.get_nowait()
                        if item is _SENTINEL:
                            render_done = True
                        else:
                            future = pool.submit(
                                self._slbr.detect_watermark_cpu, item)
                            pending_futures.append((future, item))
                            did_work = True
                    except queue.Empty:
                        pass

                # 2. Collect results in order (head-of-line)
                while pending_futures and pending_futures[0][0].done():
                    future, original_page = pending_futures.pop(0)
                    try:
                        result_page = future.result()
                        if ESRGAN_PREP_OFFLOAD:
                            result_page.esrgan_prepared = self._esrgan.prepare_tiles(
                                result_page.image_bgr)
                        # Bounded put -> blocks when SLBR_MAX_LEAD ahead
                        self._q_slbr_out.put(result_page)
                        if self._progress:
                            boxes = len(result_page.watermark_boxes)
                            status = f"p{result_page.page_idx+1}"
                            if boxes:
                                status += f" wm x{boxes}"
                            self._progress.update(Stage.SLBR, status=status)
                    except Exception as e:
                        logger.error(
                            f"SLBR CPU failed for page "
                            f"{original_page.page_idx}: {e}")
                        # Route to GPU fallback queue
                        self._q_slbr_gpu.put(original_page)
                    did_work = True

                # 3. Exit condition
                if render_done and not pending_futures:
                    break

                # 4. If nothing happened, wait a bit for render or futures
                if not did_work:
                    time.sleep(0.01)

        # Signal end of SLBR stream
        self._q_slbr_out.put(_SENTINEL)
        self._q_slbr_gpu.put(_SENTINEL)

    def _doxa_cpu_loop(self):
        """Background thread: consume NAF-DPM results, run DoxaPy, feed to Q4."""
        while True:
            item = self._q_nafdpm_out.get()
            if item is _SENTINEL:
                break

            page = item
            try:
                result_page = self._doxa.process_page(page)
                self._q_doxa_out.put(result_page)
                if self._progress:
                    self._progress.update(Stage.DOXA,
                                          status=f"p{result_page.page_idx+1}")
            except Exception as e:
                logger.error(f"DoxaPy failed for page {page.page_idx}: {e}")
                # Fallback: just convert to grayscale
                if page.nafdpm_result is not None:
                    page.final_gray = cv2.cvtColor(page.nafdpm_result, cv2.COLOR_BGR2GRAY)
                else:
                    page.final_gray = np.zeros((10, 10), dtype=np.uint8)
                page.error = f"DoxaPy: {e}"
                page.error_stage = Stage.DOXA
                self._q_doxa_out.put(page)
                if self._progress:
                    self._progress.update(Stage.DOXA, status=f"p{page.page_idx+1} err")

        self._q_doxa_out.put(_SENTINEL)

    def _gpu_main_loop(self, total_pages: int):
        """Main thread GPU controller loop.

        Cycles through stages on the single GPU thread:
        1. SLBR GPU fallback (if flagged and needed)
        2. ESRGAN (if pages available from SLBR and lead constraint allows)
        3. NAF-DPM (if patches available and batch ready)

        Lead constraints:
          SLBR.completed >= ESRGAN.completed + LEAD_PAGES
          ESRGAN.completed >= NAFDPM.completed + LEAD_PAGES

        Exits when all pages have passed through NAF-DPM.
        """
        esrgan_completed = 0
        nafdpm_pages_completed = 0

        # Pages that passed SLBR, waiting for ESRGAN (ordered)
        esrgan_queue: List[PageData] = []
        # Pages that passed ESRGAN, waiting for NAF-DPM enqueue
        nafdpm_enqueue_queue: List[PageData] = []
        while nafdpm_pages_completed < total_pages:
            did_work = False

            # --- 1. Pull available SLBR results (bounded drain) ---
            # Only pull when esrgan_queue has room. This preserves
            # _q_slbr_out (maxsize=SLBR_MAX_LEAD) backpressure so SLBR
            # stays at most SLBR_MAX_LEAD pages ahead of ESRGAN.
            while len(esrgan_queue) < LEAD_PAGES:
                try:
                    item = self._q_slbr_out.get_nowait()
                    if item is _SENTINEL:
                        self._slbr_stream_done = True
                        break
                    esrgan_queue.append(item)
                except queue.Empty:
                    break

            # --- 2. SLBR GPU fallback (process CPU-failed pages) ---
            while True:
                try:
                    gpu_item = self._q_slbr_gpu.get_nowait()
                    if gpu_item is _SENTINEL:
                        self._slbr_gpu_done = True
                        break
                    self._slbr.detect_watermark_gpu(gpu_item)
                    if ESRGAN_PREP_OFFLOAD:
                        gpu_item.esrgan_prepared = self._esrgan.prepare_tiles(gpu_item.image_bgr)
                    esrgan_queue.append(gpu_item)
                    if self._progress:
                        boxes = len(gpu_item.watermark_boxes)
                        status = f"p{gpu_item.page_idx+1} GPU"
                        if boxes:
                            status += f" wm x{boxes}"
                        self._progress.update(Stage.SLBR, status=status)
                    did_work = True
                except queue.Empty:
                    break

            # --- 3. ESRGAN: process pages if lead constraint allows ---
            while (esrgan_queue and
                   esrgan_completed - nafdpm_pages_completed < LEAD_PAGES):
                page = esrgan_queue.pop(0)
                if self._progress:
                    self._progress.set_status(Stage.ESRGAN,
                                              f"p{page.page_idx+1} tiles...")

                def _esrgan_progress(done, total):
                    if self._progress:
                        self._progress.set_status(
                            Stage.ESRGAN,
                            f"p{page.page_idx+1} {done}/{total}")

                try:
                    prepared = None
                    if ESRGAN_PREP_OFFLOAD:
                        prepared = page.esrgan_prepared
                    self._esrgan.process_page(page, progress_callback=_esrgan_progress,
                                              prepared=prepared)
                    nafdpm_enqueue_queue.append(page)
                    esrgan_completed += 1
                    if self._progress:
                        self._progress.update(Stage.ESRGAN,
                                              status=f"p{page.page_idx+1}")
                except Exception as e:
                    logger.error(f"ESRGAN failed for page {page.page_idx}: {e}")
                    page.error = f"ESRGAN: {e}"
                    page.error_stage = Stage.ESRGAN
                    # Skip this page for NAF-DPM, send directly to DoxaPy with
                    # a simple grayscale fallback
                    page.nafdpm_result = page.image_bgr
                    self._q_nafdpm_out.put(page)
                    nafdpm_pages_completed += 1
                    if self._progress:
                        self._progress.update(Stage.ESRGAN, status=f"p{page.page_idx+1} err")
                        self._progress.update(Stage.NAFDPM, status=f"p{page.page_idx+1} skip")

                did_work = True

            # --- 4. NAF-DPM: enqueue pages and process batches ---
            # Enqueue pages that finished ESRGAN (direct path)
            enqueued_this_iter = 0
            while nafdpm_enqueue_queue and (
                NAFDPM_ENQUEUE_BURST <= 0 or
                enqueued_this_iter < NAFDPM_ENQUEUE_BURST
            ):
                page = nafdpm_enqueue_queue.pop(0)
                if self._progress:
                    self._progress.set_status(Stage.NAFDPM,
                                              f"p{page.page_idx+1} init...")
                self._nafdpm.enqueue_page(page)
                enqueued_this_iter += 1

            # Process NAF-DPM batches
            if self._nafdpm.has_work():
                # Process batches until no more full batches or patches exhausted
                # Process at least one batch per loop iteration
                batch_count = 0
                while self._nafdpm.has_work():
                    # Only process full batches unless we're flushing
                    is_flushing = getattr(self, '_slbr_stream_done', False) and \
                                  not esrgan_queue and not nafdpm_enqueue_queue

                    # Deadlock break condition:
                    # ESRGAN may pause when lead gap reaches LEAD_PAGES.
                    # If NAF-DPM is waiting for a full batch at this moment,
                    # both stages can stall with no progress.
                    # In that case, allow partial-batch processing to advance NAF-DPM
                    # and reopen ESRGAN scheduling.
                    esrgan_blocked_by_lead = (
                        len(esrgan_queue) > 0 and
                        (esrgan_completed - nafdpm_pages_completed) >= LEAD_PAGES
                    )
                    allow_partial_batch = is_flushing or esrgan_blocked_by_lead

                    if not self._nafdpm.has_full_batch() and not allow_partial_batch:
                        break

                    if self._progress:
                        pending = self._nafdpm.pending_patch_count()
                        self._progress.set_status(
                            Stage.NAFDPM,
                            f"batch ({pending} patches)")

                    def _nafdpm_progress(step, total):
                        if self._progress:
                            self._progress.set_status(
                                Stage.NAFDPM,
                                f"DPM {step}/{total}")

                    completed_pages = self._nafdpm.process_batch(
                        progress_callback=_nafdpm_progress)

                    for completed_page in completed_pages:
                        self._q_nafdpm_out.put(completed_page)
                        nafdpm_pages_completed += 1
                        if self._progress:
                            self._progress.update(Stage.NAFDPM,
                                                  status=f"p{completed_page.page_idx+1}")

                    did_work = True
                    batch_count += 1

                    # Limit batches per iteration to keep ESRGAN responsive
                    if batch_count >= NAFDPM_BATCH_QUOTA_WHEN_ESRGAN_BACKLOG and esrgan_queue:
                        break

            # --- 5. Yield/wait if nothing to do ---
            if not did_work:
                time.sleep(0.01)

        # Signal end of NAF-DPM stream
        self._q_nafdpm_out.put(_SENTINEL)

    def process_document(self, pdf_path: str, page_indices: List[int],
                         dpi: int = 150) -> Dict[int, np.ndarray]:
        """Process a PDF document through the full 5-stage pipeline.

        Render -> SLBR -> ESRGAN -> NAF-DPM -> DoxaPy

        Args:
            pdf_path: Path to the PDF file
            page_indices: List of 0-based page indices to process
            dpi: Render resolution (default 150)

        Returns:
            Dict mapping page_index -> final grayscale uint8 image
        """
        total_pages = len(page_indices)
        if total_pages == 0:
            return {}

        # Create workers and queues
        self._create_workers()
        self._create_queues()
        self._slbr_stream_done = False
        self._slbr_gpu_done = False

        # Create progress bars
        self._progress = ProgressManager(total_pages)
        self._progress.create_bars()

        # Start background threads
        render_thread = threading.Thread(
            target=self._render_loop,
            args=(pdf_path, page_indices, dpi),
            name="render-loop", daemon=True)
        slbr_thread = threading.Thread(
            target=self._slbr_cpu_loop,
            name="slbr-cpu-pool", daemon=True)
        doxa_thread = threading.Thread(
            target=self._doxa_cpu_loop,
            name="doxa-cpu-pool", daemon=True)

        render_thread.start()
        slbr_thread.start()
        doxa_thread.start()

        # Run GPU main loop on current (main) thread
        self._gpu_main_loop(total_pages)

        # Collect results from DoxaPy output queue
        results: Dict[int, np.ndarray] = {}
        while True:
            item = self._q_doxa_out.get()
            if item is _SENTINEL:
                break
            page = item
            if page.final_gray is not None:
                results[page.page_idx] = page.final_gray

        # Wait for background threads
        render_thread.join(timeout=10)
        slbr_thread.join(timeout=10)
        doxa_thread.join(timeout=10)

        # Clean up
        self._progress.close()
        gc.collect()

        return results

    def process_pages(self, pages_bgr: List[Tuple[int, np.ndarray]]) -> Dict[int, np.ndarray]:
        """Backward-compatible entry point for pre-rendered pages.

        Feeds pre-rendered BGR images into the render queue directly
        (bypassing PDF rendering), then runs the rest of the pipeline.

        Args:
            pages_bgr: List of (page_index, bgr_image) tuples

        Returns:
            Dict mapping page_index -> final grayscale uint8 image
        """
        total_pages = len(pages_bgr)
        if total_pages == 0:
            return {}

        # Create workers and queues
        self._create_workers()
        self._create_queues()
        self._slbr_stream_done = False
        self._slbr_gpu_done = False

        # Create progress bars
        self._progress = ProgressManager(total_pages)
        self._progress.create_bars()

        # Feed pre-rendered pages into render queue (acts as render stage)
        def _feed_prerendered():
            for page_idx, image_bgr in pages_bgr:
                page_data = PageData(page_idx=page_idx, image_bgr=image_bgr)
                self._q_render_out.put(page_data)
                if self._progress:
                    self._progress.update(Stage.RENDER, status=f"p{page_idx+1}")
            self._q_render_out.put(_SENTINEL)

        render_thread = threading.Thread(
            target=_feed_prerendered,
            name="render-feed", daemon=True)
        slbr_thread = threading.Thread(
            target=self._slbr_cpu_loop,
            name="slbr-cpu-pool", daemon=True)
        doxa_thread = threading.Thread(
            target=self._doxa_cpu_loop,
            name="doxa-cpu-pool", daemon=True)

        render_thread.start()
        slbr_thread.start()
        doxa_thread.start()

        # Run GPU main loop on current (main) thread
        self._gpu_main_loop(total_pages)

        # Collect results from DoxaPy output queue
        results: Dict[int, np.ndarray] = {}
        while True:
            item = self._q_doxa_out.get()
            if item is _SENTINEL:
                break
            page = item
            if page.final_gray is not None:
                results[page.page_idx] = page.final_gray

        # Wait for background threads
        render_thread.join(timeout=10)
        slbr_thread.join(timeout=10)
        doxa_thread.join(timeout=10)

        # Clean up
        self._progress.close()
        gc.collect()

        return results
