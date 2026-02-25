"""
ML Document Enhancement Pipeline
Pipeline: ESRGAN (with QR protection) -> NAF-DPM -> DoxaPy 20%

GPU Acceleration: ONNX Runtime with DirectML EP (cross-vendor: NVIDIA/AMD/Intel)
Provider priority: CUDA EP -> DirectML EP -> CPU EP

All models and code are bundled locally for easy distribution.
"""
import sys
import os
import gc
from pathlib import Path
import threading
import queue
from tqdm import tqdm as _tqdm_module

# Add current directory to path for local imports
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

import cv2
import numpy as np
import time
import onnxruntime as ort
import doxapy

# Local NAF-DPM imports
from nafdpm.schedule.schedule import Schedule
from nafdpm.schedule.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver


# ============================================================
# GPU Provider Detection
# ============================================================

def detect_gpu_providers():
    """
    Detect best available ONNX Runtime execution providers.
    Priority: CUDA EP > DirectML EP > CPU EP
    
    Returns:
        tuple: (providers_list, provider_name_str)
    """
    available = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available:
        return ['CUDAExecutionProvider', 'CPUExecutionProvider'], 'CUDA'
    elif 'DmlExecutionProvider' in available:
        return ['DmlExecutionProvider', 'CPUExecutionProvider'], 'DirectML'
    else:
        return ['CPUExecutionProvider'], 'CPU'


# Module-level cached detection
_gpu_providers = None
_gpu_provider_name = None

def get_gpu_providers():
    """Get cached GPU providers (detected once)."""
    global _gpu_providers, _gpu_provider_name
    if _gpu_providers is None:
        _gpu_providers, _gpu_provider_name = detect_gpu_providers()
    return _gpu_providers, _gpu_provider_name


def create_session_options():
    """Create ONNX Runtime SessionOptions optimized for the detected provider.
    
    DirectML requires:
    - enable_mem_pattern = False (unsupported, causes slowdown if True)
    - execution_mode = ORT_SEQUENTIAL (parallel not supported)
    """
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.log_severity_level = 3  # Only show errors, suppress shape mismatch warnings
    _, provider_name = get_gpu_providers()
    if provider_name == 'DirectML':
        sess_opts.enable_mem_pattern = False
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return sess_opts


# ============================================================
# ONNX NAF-DPM Wrappers
# ============================================================

class OnnxInitPredictor:
    """ONNX wrapper for init_predictor. Accepts (B,3,128,128) numpy array batch, returns numpy array batch."""
    
    def __init__(self, session):
        self.session = session
        self.input_name = session.get_inputs()[0].name
    
    def __call__(self, batch_array):
        # batch_array: (B, 3, 128, 128) numpy array
        out = self.session.run(None, {self.input_name: batch_array})[0]  # (B,3,128,128)
        return out


class OnnxDenoiser:
    """ONNX wrapper for denoiser. Matches ConditionalNAFNet.forward(inp, time, cond) signature.
    DPM-Solver calls: model(x, t_input, cond) positionally."""
    
    def __init__(self, session):
        self.session = session
    
    def __call__(self, inp, time, cond):
        # inp: (B,3,128,128), time: scalar or (B,), cond: (B,3,128,128) - all numpy arrays
        inp_np = np.asarray(inp, dtype=np.float32)
        cond_np = np.asarray(cond, dtype=np.float32)
        # time from DPM-Solver is 0-d or (B,) numpy array
        time_arr = np.asarray(time, dtype=np.float32)
        if time_arr.ndim == 0:
            t_np = np.broadcast_to(time_arr, (inp_np.shape[0],)).copy()
        else:
            t_np = time_arr
        out = self.session.run(None, {
            'inp': inp_np, 'time': t_np, 'cond': cond_np
        })[0]
        return out



# ============================================================
# Utility functions (Numpy, general)
# ============================================================

def expand_dims(v, dims):
    """Expand tensor `v` to `dims` dimensions."""
    return v[(...,) + (None,)*(dims - 1)]

def dynamic_thresholding(x0, ratio=0.995, max_val=1.0):
    """
    The dynamic thresholding method (optimized numpy version).
    Used directly in the interleaved pipeline.
    """
    dims = x0.ndim
    p = ratio
    abs_x0 = np.abs(x0).reshape((x0.shape[0], -1))
    n_elements = abs_x0.shape[1]
    if n_elements > 16000000: # Limit size to prevent excessive memory usage or slow sort
        abs_x0 = abs_x0[:, :16000000]
        n_elements = 16000000
    k = min(int(p * n_elements), n_elements - 1)
    partitioned = np.partition(abs_x0, k, axis=1)
    s = partitioned[:, k]
    s = expand_dims(np.maximum(s, max_val * np.ones_like(s)), dims)
    x0 = np.clip(x0, -s, s) / s
    return x0




def cv2_imread_chinese(path):
    """Read image with Chinese path support"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def cv2_imwrite_chinese(path, img):
    """Write image with Chinese path support"""
    _, ext = os.path.splitext(str(path))
    success, encoded = cv2.imencode(ext, img)
    if success:
        encoded.tofile(str(path))
        return True
    return False


# ============================================================
# ESRGAN with QR protection
# ============================================================

def create_protection_mask(shape, qr_boxes, feather=30):
    """Create soft-edge mask for QR code protection"""
    h, w = shape[:2]
    mask = np.ones((h, w), dtype=np.float32)
    for x_min, y_min, x_max, y_max in qr_boxes:
        mask[y_min:y_max, x_min:x_max] = 0
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (feather*2+1, feather*2+1), feather/2)
    return mask


def apply_esrgan_tiled(image, session, tile_size=128, overlap=16, num_threads=None, progress_callback=None, return_float_rgb=False):
    """
    Apply ESRGAN with tiled processing (no upscaling).
    Pre-computes tile inputs to minimize Python overhead between GPU calls.
    Uses batch inference on CPU, sequential on DML (DML doesn't support dynamic batch for ESRGAN).
    
    Args:
        image: Input BGR image
        session: ONNX Runtime session
        tile_size: Size of each tile (default 128)
        overlap: Overlap between tiles (default 16)
        num_threads: Deprecated, kept for backward compatibility
        progress_callback: Optional callback(current, total, elapsed_sec) for progress updates
        return_float_rgb: If True, return float32 RGB [0,1] array instead of uint8 BGR.
                          Used to avoid redundant conversions when feeding into NAF-DPM.
    """
    h, w = image.shape[:2]
    input_name = session.get_inputs()[0].name
    stride = tile_size - overlap
    
    # Pre-calculate all tile coordinates
    tiles_info = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)
            tiles_info.append((y_start, y_end, x_start, x_end))
    
    total_tiles = len(tiles_info)
    start_time = time.time()
    
    # Phase 1: Pre-compute all tile inputs on CPU (avoid interleaving CPU prep with GPU calls)
    tile_inputs = []  # list of (3, 128, 128) numpy arrays
    pad_infos = []    # (pad_h, pad_w) per tile
    for y_start, y_end, x_start, x_end in tiles_info:
        tile = image[y_start:y_end, x_start:x_end]
        pad_h = tile_size - tile.shape[0]
        pad_w = tile_size - tile.shape[1]
        if pad_h > 0 or pad_w > 0:
            tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        pad_infos.append((pad_h, pad_w))
        tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tile_inputs.append(np.transpose(tile_rgb, (2, 0, 1)))  # (3, 128, 128)
    
    # Phase 2: Inference
    providers = [p if isinstance(p, str) else p[0] for p in session.get_providers()]
    use_dml_threaded = 'DmlExecutionProvider' in providers # Use threading for DirectML
    
    output = np.zeros_like(image, dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    
    def _post_process_tile(raw_result, idx, tile_size, return_float_rgb, pad_infos, tiles_info):
        """Post-process a single tile result and return for accumulation."""
        result = np.transpose(raw_result, (1, 2, 0))  # (H, W, 3) RGB float32 [0,1]
        result = np.clip(result, 0, 1)
        if result.shape[0] != tile_size: # Should not happen with 4x upscale and 128->512 tile
            result = cv2.resize(result, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        if return_float_rgb:
            # Keep float32 RGB — skip uint8/BGR conversion entirely
            tile_out = result
        else:
            tile_out = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR).astype(np.float32)
        pad_h, pad_w = pad_infos[idx]
        if pad_h > 0:
            tile_out = tile_out[:-pad_h]
        if pad_w > 0:
            tile_out = tile_out[:, :-pad_w]
        y_start, y_end, x_start, x_end = tiles_info[idx]
        # Return the processed tile and its coordinates for later accumulation
        return tile_out, (y_start, y_end, x_start, x_end)
    
    def _post_process_tile_wrapper(raw_result, idx):
        return _post_process_tile(raw_result, idx, tile_size, return_float_rgb, pad_infos, tiles_info)

    if not use_dml_threaded: # Original batch processing for CPU EP or other EPs that support batching
        # CPU: batch multiple tiles together (3-5x speedup)
        batch_size = 8
        for batch_start in range(0, total_tiles, batch_size):
            batch_end = min(batch_start + batch_size, total_tiles)
            batch = np.stack(tile_inputs[batch_start:batch_end], axis=0)  # (N, 3, 128, 128)
            results = session.run(None, {input_name: batch})[0]  # (N, 3, H_out, W_out)
            for j in range(batch_end - batch_start):
                processed_tile, coords = _post_process_tile_wrapper(results[j], batch_start + j)
                y_start, y_end, x_start, x_end = coords
                output[y_start:y_end, x_start:x_end] += processed_tile
                weight[y_start:y_end, x_start:x_end] += 1
            if progress_callback:
                elapsed = time.time() - start_time
                progress_callback(batch_end, total_tiles, elapsed)
    else:
        # DML (DirectML): 3-thread pipeline: Producer (CPU) -> GPU Worker (DML) -> Consumer (CPU)
        # Use IO Binding to reduce allocation overhead
        input_tensor_shape_raw = session.get_inputs()[0].shape
        output_tensor_shape_raw = session.get_outputs()[0].shape
        
        # Convert symbolic dimensions to concrete values (batch=1 for tile processing)
        input_tensor_shape = tuple(1 if isinstance(dim, str) else dim for dim in input_tensor_shape_raw)
        output_tensor_shape = tuple(1 if isinstance(dim, str) else dim for dim in output_tensor_shape_raw)
        
        input_dtype = np.float32 # ESRGAN model expects float32
        output_dtype = np.float32 # ESRGAN model outputs float32

        # Pre-allocate input and output buffers on CPU (DirectML copies these to GPU memory)
        input_buffer = np.empty(input_tensor_shape, dtype=input_dtype)
        output_buffer = np.empty(output_tensor_shape, dtype=output_dtype)

        io_binding = session.io_binding()
        
        io_binding.bind_input(
            name=input_name,
            device_type='cpu', # DirectML transfers from CPU to GPU
            device_id=0,
            element_type=input_dtype,
            shape=input_tensor_shape,
            buffer_ptr=input_buffer.ctypes.data
        )
        io_binding.bind_output(
            name=session.get_outputs()[0].name,
            device_type='cpu', # DirectML transfers from GPU to CPU
            device_id=0,
            element_type=output_dtype,
            shape=output_tensor_shape,
            buffer_ptr=output_buffer.ctypes.data
        )

        # Queues for producer-consumer pattern
        # Maxsize chosen to allow some buffering without excessive memory usage
        gpu_input_queue = queue.Queue(maxsize=4) 
        processed_output_queue = queue.Queue(maxsize=4)
        
        # Thread communication / termination signals
        termination_event = threading.Event()
        # Lock for accumulating results into shared 'output' and 'weight' arrays
        accumulation_lock = threading.Lock()
        
        # Keep track of completed tiles for progress reporting
        tiles_processed_by_consumer = 0
        
        def producer_thread_func():
            try:
                for idx in range(total_tiles):
                    if termination_event.is_set():
                        break
                    # Put pre-computed tile input (already np.float32 (3,128,128)) and its index
                    gpu_input_queue.put((tile_inputs[idx], idx)) 
            finally:
                gpu_input_queue.put(None) # Sentinel to signal end of production

        def gpu_worker_thread_func():
            try:
                while True:
                    if termination_event.is_set():
                        break
                    
                    item = gpu_input_queue.get()
                    if item is None:
                        break # End of production
                    
                    tile_input_np, idx = item
                    
                    # Copy tile input to pre-allocated buffer
                    np.copyto(input_buffer, tile_input_np[np.newaxis, ...])
                    
                    # Run inference with IO binding
                    session.run_with_iobinding(io_binding)
                    
                    # Retrieve result from the output buffer
                    # MUST copy here, as the output_buffer is reused by next GPU call
                    raw_result = output_buffer.copy() 
                    processed_output_queue.put((raw_result[0], idx))
            finally:
                processed_output_queue.put(None) # Sentinel for consumer
        
        def consumer_thread_func():
            nonlocal tiles_processed_by_consumer
            try:
                while True:
                    if termination_event.is_set():
                        break
                    
                    item = processed_output_queue.get()
                    if item is None:
                        break # End of processing
                    
                    raw_result_tile, idx = item
                    
                    # Post-process and accumulate (using the refactored function)
                    processed_tile, coords = _post_process_tile_wrapper(raw_result_tile, idx)
                    
                    with accumulation_lock:
                        y_start, y_end, x_start, x_end = coords
                        output[y_start:y_end, x_start:x_end] += processed_tile
                        weight[y_start:y_end, x_start:x_end] += 1
                        tiles_processed_by_consumer += 1
                        
                        if progress_callback:
                            elapsed = time.time() - start_time
                            progress_callback(tiles_processed_by_consumer, total_tiles, elapsed)
            finally:
                pass # No more items to put or signal

        # Start threads
        producer = threading.Thread(target=producer_thread_func)
        gpu_worker = threading.Thread(target=gpu_worker_thread_func)
        consumer = threading.Thread(target=consumer_thread_func)
        
        producer.start()
        gpu_worker.start()
        consumer.start()
        
        # Wait for all threads to complete
        producer.join()
        gpu_worker.join()
        consumer.join()

        # Clear IO Binding after use
        io_binding.clear_binding_inputs()
        io_binding.clear_binding_outputs()


    
    output = output / np.maximum(weight[:, :, np.newaxis], 1)
    if return_float_rgb:
        return output  # float32 RGB [0,1]
    return output.astype(np.uint8)


# ============================================================
# NAF-DPM inference
# ============================================================

def load_nafdpm_onnx(models_dir, device='cpu'):
    """
    Load NAF-DPM via ONNX Runtime (GPU-accelerated).
    
    Returns:
        tuple: (init_predictor_callable, denoiser_callable, schedule)
        - init_predictor_callable: accepts (B,3,128,128) numpy array, returns (B,3,128,128) numpy array
        - denoiser_callable: accepts (inp, time, cond) numpy arrays, returns numpy array
        - schedule: Schedule object for DPM-Solver
    """
    models_dir = Path(models_dir)
    onnx_init_path = models_dir / "NAF-DPM_init_predictor-fp16.onnx"
    onnx_denoiser_path = models_dir / "NAF-DPM_denoiser-fp16.onnx"
    
    timesteps = 100
    schedule = Schedule('linear', timesteps)
    
    if onnx_init_path.exists() and onnx_denoiser_path.exists():
        providers, provider_name = get_gpu_providers()
        sess_opts = create_session_options()
        
        init_session = ort.InferenceSession(
            str(onnx_init_path), sess_opts, providers=providers
        )
        denoiser_session = ort.InferenceSession(
            str(onnx_denoiser_path), sess_opts, providers=providers
        )
        
        init_predictor = OnnxInitPredictor(init_session)
        denoiser = OnnxDenoiser(denoiser_session)
        
        return init_predictor, denoiser, schedule
    
    raise FileNotFoundError(
        f"NAF-DPM ONNX模型文件未找到: {onnx_init_path} 或 {onnx_denoiser_path}"
    )


def nafdpm_crop_concat(img_array, size=128):
    """Split image array into 128x128 patches for native resolution processing"""
    shape = img_array.shape
    h, w = shape[2], shape[3]
    
    n_h = (h + size - 1) // size
    n_w = (w + size - 1) // size
    correct_h = n_h * size
    correct_w = n_w * size
    
    padded = np.ones((shape[0], shape[1], correct_h, correct_w), dtype=np.float32)
    padded[:, :, :h, :w] = img_array
    
    patches = []
    for i in range(n_h):
        for j in range(n_w):
            patch = padded[:, :, i*size:(i+1)*size, j*size:(j+1)*size]
            patches.append(patch)
    
    return np.concatenate(patches, axis=0), (n_h, n_w, h, w)


def nafdpm_crop_concat_back(patches, grid_info, size=128):
    """Reconstruct image from patches"""
    n_h, n_w, orig_h, orig_w = grid_info
    
    rows = []
    for i in range(n_h):
        row_patches = []
        for j in range(n_w):
            idx = i * n_w + j
            row_patches.append(patches[idx:idx+1])
        rows.append(np.concatenate(row_patches, axis=3))
    
    full = np.concatenate(rows, axis=2)
    return full[:, :, :orig_h, :orig_w]


def create_dpm_solver_context(schedule, denoiser):
    """Create reusable DPM-Solver context (noise schedule + denoiser binding).
    
    Returns a function that can be called repeatedly with different noisy_image/condition
    without recreating NoiseScheduleVP/model_wrapper/DPM_Solver each time.
    """
    betas = schedule.get_betas()
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)
    return noise_schedule, denoiser


def _get_dpm_model_fn(denoiser, noise_schedule, condition):
    """
    Helper to create the model_fn (denoiser wrapper) for DPM-Solver.
    Extracted from DPM_Solver.sample() to be reusable in interleaved pipeline.
    """
    model_fn = model_wrapper(
        denoiser,
        noise_schedule,
        model_type="x_start",
        model_kwargs={},
        guidance_type="classifier-free",
        condition=condition
    )
    return model_fn


def dpm_solver_sample(schedule_or_context, denoiser_or_none, noisy_image, dpm_steps, condition):
    """Run DPM-Solver sampling.
    
    Args:
        schedule_or_context: Either a Schedule object (legacy) or a (noise_schedule, denoiser) tuple (cached).
        denoiser_or_none: Denoiser callable if schedule_or_context is a Schedule, else None.
    """
    if isinstance(schedule_or_context, tuple):
        # Cached path — reuse pre-built noise_schedule
        noise_schedule, denoiser = schedule_or_context
    else:
        # Legacy path — build from scratch
        betas = schedule_or_context.get_betas()
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)
        denoiser = denoiser_or_none
    
    model_fn = _get_dpm_model_fn(denoiser, noise_schedule, condition)
    
    dpm_solver = DPM_Solver(
        model_fn, noise_schedule,
        algorithm_type="dpmsolver++",
        correcting_x0_fn="dynamic_thresholding"
    )
    
    x_sample = dpm_solver.sample(
        noisy_image,
        steps=dpm_steps,
        order=1,
        skip_type="time_uniform",
        method="singlestep",
    )
    return x_sample


def apply_nafdpm(image, init_predictor, denoiser, schedule, device='cpu',
                 dpm_steps=20, batch_size=8, progress_callback=None,
                 input_float_rgb=False, _rng_seed=None):
    """
    Apply NAF-DPM document enhancement with step-level interleaved pipeline.
    
    Interleaving pattern: B1_step1_GPU → B2_step1_GPU (overlap B1_step1_CPU) → ...
    All GPU calls on main thread; CPU work (thresholding + solver arithmetic)
    runs on a worker thread, overlapping with the next batch's GPU call.
    
    Args:
        progress_callback: Optional callback(completed_batches, total_batches, elapsed_sec)
    """
    start_time = time.time()
    
    if input_float_rgb:
        image_rgb = image
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    img_array = np.transpose(image_rgb, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
    
    patches, grid_info = nafdpm_crop_concat(img_array, size=128)
    num_patches = patches.shape[0]
    num_batches = (num_patches + batch_size - 1) // batch_size
    
    # Build noise schedule
    noise_schedule, _ = create_dpm_solver_context(schedule, denoiser)
    ns = noise_schedule
    total_N = ns.total_N  # 100
    
    # Pre-compute init_predictor for all batches (GPU, sequential)
    init_predicts = []
    x_states = []
    _rng = np.random.default_rng(_rng_seed)
    for b_idx in range(num_batches):
        b_start = b_idx * batch_size
        b_end = min(b_start + batch_size, num_patches)
        batch = patches[b_start:b_end]
        init_pred = init_predictor(batch)  # GPU call
        init_predicts.append(init_pred)
        # Initial noisy state
        x_states.append(_rng.standard_normal(
            init_pred.shape).astype(np.float32))
    del patches
    
    # Pre-compute timesteps (singlestep, order=1, time_uniform)
    t_T = ns.T
    t_0 = 1. / total_N
    timesteps = np.linspace(t_T, t_0, dpm_steps + 1).astype(np.float32)
    
    # Step-level interleaved DPM loop
    # (Validated in test_interleave.py: 0.00 max_diff vs sequential, 5.3% faster)
    cpu_thread = None
    cpu_result = [None, None]  # [batch_idx, x_t]
    
    completed_batches = 0  # for progress tracking
    total_batch_ops = num_batches * dpm_steps  # total GPU calls
    
    for step in range(dpm_steps):
        s_val = timesteps[step]
        t_val = timesteps[step + 1]
        
        # Pre-compute schedule values (CPU, <0.1ms, shared across batches)
        lambda_s = ns.marginal_lambda(s_val)
        lambda_t = ns.marginal_lambda(t_val)
        h = lambda_t - lambda_s
        log_alpha_t = ns.marginal_log_mean_coeff(t_val)
        sigma_s = ns.marginal_std(s_val)
        sigma_t = ns.marginal_std(t_val)
        alpha_t = np.exp(log_alpha_t)
        phi_1 = np.expm1(-h)
        
        # Model input time: discrete schedule conversion
        t_input = float((s_val - 1. / total_N) * 1000.)
        
        for b in range(num_batches):
            x = x_states[b]
            cond = init_predicts[b]
            
            # === GPU: denoiser call (main thread) ===
            t_arr = np.broadcast_to(np.float32(t_input), (x.shape[0],)).copy()
            x0_pred = denoiser(x, t_arr, cond)
            
            # Wait for previous batch's CPU work before overwriting its state
            if cpu_thread is not None:
                cpu_thread.join()
                x_states[cpu_result[0]] = cpu_result[1]
            
            # === CPU: thresholding + solver arithmetic (worker thread) ===
            def _cpu_step(_b, _x, _x0, _ss, _st, _at, _p1):
                x0_c = dynamic_thresholding(_x0)
                x_t = (_st / _ss) * _x - _at * _p1 * x0_c
                cpu_result[0] = _b
                cpu_result[1] = x_t
            
            cpu_thread = threading.Thread(
                target=_cpu_step,
                args=(b, x, x0_pred, sigma_s, sigma_t, alpha_t, phi_1))
            cpu_thread.start()
            
            # Progress callback (compatible with existing callers)
            completed_batches += 1
            if progress_callback:
                elapsed = time.time() - start_time
                progress_callback(completed_batches, total_batch_ops, elapsed)
    
    # Join final CPU thread
    if cpu_thread is not None:
        cpu_thread.join()
        x_states[cpu_result[0]] = cpu_result[1]
    
    # Final: clip and reconstruct
    result_patches = []
    for b in range(num_batches):
        result_patches.append(np.clip(x_states[b] + init_predicts[b], 0, 1))
    del x_states, init_predicts
    
    all_patches = np.concatenate(result_patches, axis=0)
    del result_patches
    
    reconstructed = nafdpm_crop_concat_back(all_patches, grid_info, size=128)
    del all_patches
    
    # Convert back to BGR uint8
    result_np = np.transpose(reconstructed[0], (1, 2, 0))
    del reconstructed
    result_np = np.clip(result_np * 255, 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
    del result_np
    
    gc.collect()
    return result_bgr


# ============================================================
# DoxaPy whitening
# ============================================================

def apply_doxa(image, blend_ratio=0.20):
    """Apply DoxaPy Sauvola binarization with blending"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    binary = np.zeros_like(gray, dtype=np.uint8)
    binarizer = doxapy.Binarization(doxapy.Binarization.Algorithms.SAUVOLA)
    binarizer.initialize(gray)
    binarizer.to_binary(binary)
    blended = cv2.addWeighted(gray, 1 - blend_ratio, binary, blend_ratio, 0)
    if len(image.shape) == 3:
        blended = cv2.cvtColor(blended, cv2.COLOR_GRAY2BGR)
    return blended


# ============================================================
# Main Pipeline Class
# ============================================================

class MLDocumentEnhancer:
    """
    ML-based Document Enhancement Pipeline
    Pipeline: ESRGAN (QR protected) -> NAF-DPM -> DoxaPy 20%
    """
    
    def __init__(self, models_dir=None, device='cpu'):
        """
        Initialize the enhancer with model paths.
        
        Args:
            models_dir: Directory containing models. Defaults to ./models
            device: 'cpu' or 'cuda'
        """
        if models_dir is None:
            models_dir = SCRIPT_DIR / "models"
        else:
            models_dir = Path(models_dir)
        
        self.device = device
        self.models_dir = models_dir
        
        # Model paths
        self.esrgan_path = models_dir / "real-esrgan-x4plus-128-fp16.onnx"
        self.nafdpm_onnx_init_path = models_dir / "NAF-DPM_init_predictor-fp16.onnx"
        self.nafdpm_onnx_denoiser_path = models_dir / "NAF-DPM_denoiser-fp16.onnx"
        self.slbr_path = models_dir / "slbr-watermark-detector-fp16.onnx"
        
        # Lazy loading
        self._esrgan_session = None
        self._esrgan_provider_info = None
        self._nafdpm_init_predictor = None  # callable
        self._nafdpm_denoiser = None        # callable
        self._nafdpm_schedule = None
        self._nafdpm_provider_info = None
        self._slbr_session = None
        self._slbr_provider_info = None
    
    def _load_esrgan(self, progress_callback=None, current_stage=None):
        """Load ESRGAN model (lazy) with GPU acceleration"""
        if self._esrgan_session is None:
            if progress_callback and current_stage is not None:
                progress_callback("ESRGAN", current_stage, "加载模型...", 0.0)
            providers, provider_name = get_gpu_providers()
            sess_opts = create_session_options()
            self._esrgan_session = ort.InferenceSession(
                str(self.esrgan_path), sess_opts,
                providers=providers
            )
            actual_provider = self._esrgan_session.get_providers()[0]
            self._esrgan_provider_info = f"ONNX Runtime ({actual_provider})"
            if progress_callback and current_stage is not None:
                progress_callback("ESRGAN", current_stage, f"模型已加载 ({self._esrgan_provider_info})", 0.0)
        return self._esrgan_session
    
    def _load_nafdpm(self, progress_callback=None, current_stage=None):
        """Load NAF-DPM model (lazy) — ONNX Runtime with GPU acceleration"""
        if self._nafdpm_init_predictor is None:
            if progress_callback and current_stage is not None:
                progress_callback("NAF-DPM", current_stage, "加载模型...", 0.0)
            self._nafdpm_init_predictor, self._nafdpm_denoiser, self._nafdpm_schedule = \
                load_nafdpm_onnx(self.models_dir, self.device)
            actual = self._nafdpm_init_predictor.session.get_providers()[0]
            self._nafdpm_provider_info = f"ONNX Runtime ({actual})"
            if progress_callback and current_stage is not None:
                progress_callback("NAF-DPM", current_stage, f"模型已加载 ({self._nafdpm_provider_info})", 0.0)
        return self._nafdpm_init_predictor, self._nafdpm_denoiser, self._nafdpm_schedule
    
    def _load_slbr(self):
        """Load SLBR watermark detector ONNX model (lazy) with GPU acceleration"""
        if self._slbr_session is None:
            if not self.slbr_path.exists():
                return None
            providers, provider_name = get_gpu_providers()
            sess_opts = create_session_options()
            self._slbr_session = ort.InferenceSession(
                str(self.slbr_path), sess_opts,
                providers=providers
            )
            actual_provider = self._slbr_session.get_providers()[0]
            self._slbr_provider_info = f"ONNX Runtime ({actual_provider})"
        return self._slbr_session
    
    def detect_watermark(self, image_bgr, tile_size=256, overlap=64, threshold=0.3, min_area=2000, progress_callback=None):
        """
        Detect watermark regions using SLBR ONNX model with tiled inference.
        
        Args:
            image_bgr: Input BGR image (numpy array)
            tile_size: Tile size for SLBR inference (default 256)
            overlap: Overlap between tiles (default 64)
            threshold: Mask binarization threshold (default 0.3)
            min_area: Minimum contour area to consider as watermark (default 2000)
            progress_callback: Optional callback(info_dict) for progress updates
            
        Returns:
            List of (x_min, y_min, x_max, y_max) bounding boxes for watermark regions,
            or empty list if no watermarks detected or model not available.
        """
        session = self._load_slbr()
        if session is None:
            return []
        
        h, w = image_bgr.shape[:2]
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        input_name = session.get_inputs()[0].name
        
        # Accumulate mask with weighted averaging for overlapping tiles
        mask_sum = np.zeros((h, w), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)
        stride = tile_size - overlap
        
        # Pre-calculate tile coordinates
        tiles = []
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                y_start = max(0, y_end - tile_size)
                x_start = max(0, x_end - tile_size)
                tiles.append((y_start, y_end, x_start, x_end))
        
        total_tiles = len(tiles)
        start_time = time.time()
        
        for i, (y_start, y_end, x_start, x_end) in enumerate(tiles):
            tile = img_rgb[y_start:y_end, x_start:x_end]
            
            pad_h = tile_size - tile.shape[0]
            pad_w = tile_size - tile.shape[1]
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
            
            if progress_callback and (i + 1) % 5 == 0:
                elapsed = time.time() - start_time
                remaining = (elapsed / (i + 1)) * (total_tiles - i - 1)
                progress_callback({
                    'stage': 'SLBR',
                    'detail': f"tiles {i+1}/{total_tiles} ({elapsed:.1f}s/剩余~{remaining:.0f}s)",
                    'progress': (i + 1) / total_tiles
                })
        
        mask_avg = mask_sum / np.maximum(weight_sum, 1e-6)
        
        # Extract bounding boxes from mask
        mask_uint8 = (mask_avg * 255).clip(0, 255).astype(np.uint8)
        _, binary = cv2.threshold(mask_uint8, int(threshold * 255), 255, cv2.THRESH_BINARY)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        watermark_boxes = []
        margin = 30  # padding around detected region
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch
            aspect_ratio = float(cw) / ch if ch > 0 else 0
            
            if area > min_area and 0.3 < aspect_ratio < 3.0:
                x_min = max(0, x - margin)
                y_min = max(0, y - margin)
                x_max = min(w, x + cw + margin)
                y_max = min(h, y + ch + margin)
                watermark_boxes.append((x_min, y_min, x_max, y_max))
        
        return watermark_boxes
    
    def enhance(self, image_bgr, qr_boxes=None, dpm_steps=20, doxa_blend=0.20,
                skip_esrgan=False, skip_nafdpm=False, skip_doxa=False,
                progress_callback=None, silent=True,
                esrgan_overlap=8, nafdpm_batch_size=32):
        """
        Run the full enhancement pipeline on a BGR image.
        
        Args:
            image_bgr: Input BGR image (numpy array)
            qr_boxes: List of (x_min, y_min, x_max, y_max) for QR code protection
            dpm_steps: Number of DPM-Solver steps (default 20 for best quality)
            doxa_blend: DoxaPy blend ratio (default 0.20)
            skip_esrgan: Skip ESRGAN step
            skip_nafdpm: Skip NAF-DPM step
            skip_doxa: Skip DoxaPy step
            progress_callback: Optional callback(step_info_dict) for progress updates
                step_info_dict = {
                    'stage': 'ESRGAN' | 'NAF-DPM' | 'DoxaPy',
                    'stage_num': 1 | 2 | 3,
                    'detail': str,  # e.g., "tiles 15/100 (12.5s)"
                    'progress': float  # 0.0 to 1.0
                }
            silent: If True, suppress print output
            esrgan_overlap: ESRGAN tile overlap (default 8, optimized for DirectML)
            nafdpm_batch_size: NAF-DPM batch size (default 32, optimal for ~128 patches per page;
                batch 20-64 is a performance plateau, 32 is the sweet spot with lowest ms/patch)
        
        Returns:
            Enhanced BGR image
        """
        result = image_bgr.copy()
        total_stages = 3 - int(skip_esrgan) - int(skip_nafdpm) - int(skip_doxa)
        current_stage = 0
        
        # Optimization: when both ESRGAN and NAF-DPM run, pass float32 RGB directly
        # to avoid redundant uint8→BGR→float32→RGB conversions at handoff
        use_float_rgb_handoff = not skip_esrgan and not skip_nafdpm
        
        def log(msg):
            if not silent:
                print(msg)
        
        def notify_progress(stage, stage_num, detail="", progress=0.0):
            if progress_callback:
                progress_callback({
                    'stage': stage,
                    'stage_num': stage_num,
                    'total_stages': total_stages,
                    'detail': detail,
                    'progress': progress
                })
        
        # Step 1: ESRGAN with QR protection (multi-threaded)
        if not skip_esrgan:
            current_stage += 1
            log("[步骤1] ESRGAN 超分辨率...")
            
            session = self._load_esrgan(notify_progress, current_stage)
            
            # GPU (DML/CUDA) sessions are not thread-safe; disable multi-threading
            _, provider_name = get_gpu_providers()
            esrgan_threads = 1 if provider_name != 'CPU' else None
            
            # ESRGAN tile progress callback
            def esrgan_tile_progress(completed, total, elapsed):
                pct = completed / total
                remaining = (elapsed / max(completed, 1)) * (total - completed) if completed > 0 else 0
                detail = f"tiles {completed}/{total} ({elapsed:.1f}s/剩余~{remaining:.0f}s)"
                notify_progress("ESRGAN", current_stage, detail, pct)
            
            enhanced = apply_esrgan_tiled(
                result, session, tile_size=128, overlap=esrgan_overlap,
                num_threads=esrgan_threads,
                progress_callback=esrgan_tile_progress,
                return_float_rgb=use_float_rgb_handoff
            )
            
            # Apply QR protection mask
            if qr_boxes:
                protection_mask = create_protection_mask(result.shape, qr_boxes, feather=30)
                mask_3ch = protection_mask[:, :, np.newaxis]
                if use_float_rgb_handoff:
                    # Blend in float32 RGB space — convert original to match
                    original_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    result = enhanced * mask_3ch + original_rgb * (1 - mask_3ch)
                else:
                    result = (enhanced * mask_3ch + result * (1 - mask_3ch)).astype(np.uint8)
            else:
                result = enhanced
            
            notify_progress("ESRGAN", current_stage, "完成", 1.0)
        
        # Step 2: NAF-DPM enhancement
        if not skip_nafdpm:
            current_stage += 1
            log("[步骤2] NAF-DPM 文档增强...")
            
            init_predictor, denoiser, schedule = self._load_nafdpm(notify_progress, current_stage)
            
            # NAF-DPM progress callback
            def nafdpm_batch_progress(completed, total, elapsed):
                pct = completed / total
                remaining = (elapsed / max(completed, 1)) * (total - completed) if completed > 0 else 0
                detail = f"DPM {completed}/{total} ({elapsed:.1f}s/剩余~{remaining:.0f}s)"
                notify_progress("NAF-DPM", current_stage, detail, pct)
            
            result = apply_nafdpm(
                result, init_predictor, denoiser, schedule,
                self.device, dpm_steps=dpm_steps, batch_size=nafdpm_batch_size,
                progress_callback=nafdpm_batch_progress,
                input_float_rgb=use_float_rgb_handoff
            )
            
            notify_progress("NAF-DPM", current_stage, "完成", 1.0)
        
        # Step 3: DoxaPy whitening
        if not skip_doxa:
            current_stage += 1
            notify_progress("DoxaPy", current_stage, "二值化混合...", 0.5)
            log("[步骤3] DoxaPy 白化...")
            
            result = apply_doxa(result, blend_ratio=doxa_blend)
            
            notify_progress("DoxaPy", current_stage, "完成", 1.0)
        
        return result
    
    def enhance_file(self, input_path, output_path, **kwargs):
        """
        Enhance an image file.
        
        Args:
            input_path: Input image path
            output_path: Output image path
            **kwargs: Arguments passed to enhance()
        """
        image = cv2_imread_chinese(input_path)
        if image is None:
            raise ValueError(f"Cannot read image: {input_path}")
        
        result = self.enhance(image, **kwargs)
        cv2_imwrite_chinese(output_path, result)
        return result


# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Document Enhancement Pipeline")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", help="Output image path")
    parser.add_argument("--models-dir", help="Models directory (default: ./models)")
    parser.add_argument("--dpm-steps", type=int, default=20, help="DPM-Solver steps (default: 20)")
    parser.add_argument("--doxa-blend", type=float, default=0.20, help="DoxaPy blend ratio (default: 0.20)")
    parser.add_argument("--qr-box", type=str, help="QR box: x_min,y_min,x_max,y_max")
    parser.add_argument("--skip-esrgan", action="store_true", help="Skip ESRGAN step")
    parser.add_argument("--skip-nafdpm", action="store_true", help="Skip NAF-DPM step")
    parser.add_argument("--skip-doxa", action="store_true", help="Skip DoxaPy step")
    
    args = parser.parse_args()
    
    # Parse QR box
    qr_boxes = None
    if args.qr_box:
        parts = [int(x) for x in args.qr_box.split(",")]
        qr_boxes = [tuple(parts)]
    
    # Create enhancer and run
    enhancer = MLDocumentEnhancer(models_dir=args.models_dir)
    enhancer.enhance_file(
        args.input, args.output,
        qr_boxes=qr_boxes,
        dpm_steps=args.dpm_steps,
        doxa_blend=args.doxa_blend,
        skip_esrgan=args.skip_esrgan,
        skip_nafdpm=args.skip_nafdpm,
        skip_doxa=args.skip_doxa
    )
    print(f"Done! Output saved to: {args.output}")


if __name__ == "__main__":
    main()
