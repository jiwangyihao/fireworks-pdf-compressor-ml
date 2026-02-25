# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for Fireworks PDF Compressor with ML Enhancement
# Usage: 
#   1. 安装GPU加速的onnxruntime:
#      pip install onnxruntime-directml
#   2. 打包:
#      pyinstaller compress.spec
#   注意: 已完全移除PyTorch依赖，使用纯numpy实现DPM-Solver

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# 隐式导入
hiddenimports = [
    'onnxruntime',
    'cv2',
    'numpy',
    'PIL',
    'PIL.Image',
    'doxapy',
    'pikepdf',
    'fitz',
    'tqdm',
    # imagecodecs 使用延迟导入，PyInstaller 静态分析无法检测
    'imagecodecs',
    'imagecodecs.imagecodecs',
    'imagecodecs._jpeg2k',
    'imagecodecs._shared',
    'imagecodecs._shared_cython',
]

# 收集 onnxruntime-directml DLL 路径 (DirectML.dll等)
import onnxruntime as _ort
_ort_capi_dir = os.path.join(os.path.dirname(_ort.__file__), 'capi')
_ort_binaries = []
for _dll in ['DirectML.dll', 'onnxruntime.dll', 'onnxruntime_providers_shared.dll']:
    _dll_path = os.path.join(_ort_capi_dir, _dll)
    if os.path.exists(_dll_path):
        _ort_binaries.append((_dll_path, '.'))

a = Analysis(
    ['compress.py'],
    pathex=[],
    binaries=_ort_binaries,
    datas=[
        # 模型文件 (FP16 ONNX，体积减半，速度提升~40%)
        ('models/real-esrgan-x4plus-128-fp16.onnx', 'models'),
        ('models/NAF-DPM_init_predictor-fp16.onnx', 'models'),
        ('models/NAF-DPM_denoiser-fp16.onnx', 'models'),
        # SLBR 水印检测模型 (FP16 ONNX, 31.6MB)
        ('models/slbr-watermark-detector-fp16.onnx', 'models'),
        # NAF-DPM 源码
        ('nafdpm', 'nafdpm'),
        # ML增强脚本
        ('ml_enhance.py', '.'),
        # ML流水线 (4阶段生产者-消费者管线)
        ('ml_pipeline.py', '.'),
        # 自适应配置
        ('adaptive_config.py', '.'),
        # Ghostscript (如果需要)
        ('gs', 'gs'),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 彻底排除PyTorch（已不再使用）
        'torch',
        'torch.nn',
        'torch.nn.functional',
        'torchvision',
        'torchaudio',
        # 排除不需要的模块以减小体积
        'matplotlib',
        'tkinter',
        'PyQt5',
        'PySide2',
        'IPython',
        'jupyter',
        'notebook',
        # 排除TensorFlow
        'tensorflow',
        'tensorflow_core',
        'keras',
        'tensorboard',
        # 排除其他大型不需要的包
        'transformers',
        'numba',
        'llvmlite',
        'pytest',
        'pandas',
        'h5py',
        'grpc',
        'grpcio',
        'pydantic',
        'uvicorn',
        'starlette',
        'fastapi',
        'anyio',
        'einops',
        # scipy不被直接使用
        'scipy',
        'scipy.special',
        'scipy.linalg',
        'scipy.sparse',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='FireworksPDFCompressor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # 保持控制台窗口以显示进度
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # 可以添加 icon='icon.ico'
)
