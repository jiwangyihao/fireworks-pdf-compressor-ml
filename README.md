# Fireworks PDF Compressor ML

[![License: MPL-2.0](https://img.shields.io/badge/License-MPL%202.0-orange.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Dependency Manager: uv](https://img.shields.io/badge/deps-uv-7c3aed.svg)](https://docs.astral.sh/uv/)

中文 | [English](#english)

面向 GitHub 仓库托管场景的 PDF 优化工具，目标是在保持可读性的前提下，将大体积 PDF 尽可能压缩到 **100MB 以下**。

---

## 中文

### 项目目标

- 服务于 GitHub 仓库中的 PDF 分发场景（100MB 限制）。
- 在压缩率与可读性之间提供工程化平衡。
- 支持可选 ML 图像增强链路与 Windows EXE 发布。

### 处理流水线

- **阶段一（无损优先）**
  1. 结构清理（对象与元数据层面）
  2. Ghostscript 重排（含文本异常检测与回退）
  3. 安全图片压缩（低风险重编码）
- **阶段二（有损交错）**
  - 固定顺序：`矢量 -> 图片 -> 切片`
  - 固定强度阶梯：`50dB -> 45dB -> 40dB -> 35dB`
  - 文件仍超目标时逐步增强压缩强度

### 矢量压缩

- 矢量压缩作用于 PDF 绘图指令流（路径命令与坐标数字）。
- 强度从低到高可理解为四个层级：
  - **L1（保守）**：仅轻度减少坐标冗余，视觉变化通常不可察觉，适合对版式稳定性要求极高的文档。
  - **L2（均衡）**：进一步降低路径数值精度，在大多数课件和笔记中能获得更好的体积收益，同时保持线条连续性。
  - **L3（激进）**：明显加强路径压缩，对复杂矢量页面收益更高；在局部高密度细节处，线条圆滑度可能轻微下降。
  - **L4（最强）**：在 L3 基础上进一步简化极短曲线段，优先追求极限体积；更适合以“可读性优先、细节次之”的分发场景。
- 实际执行时，这些强度会在有损阶段中逐步推进，而不是一次性直接跳到最高强度。

### 文件类型与收益侧重点

- **板写笔记 / 课件（矢量元素多）**：矢量与结构压缩收益更明显。
- **扫描件 / 图片型 PDF**：图片重编码与切片压缩收益更明显。
- **灰度页面**：文本区域可激进二值化；灰度细节区域可通过混合切片保留灰度信息，避免整页硬二值化导致细节损失。

- **编码策略**
  - 二值数据：`FlateDecode`
  - 常规图像：`JPXDecode`（JPEG2000）

- **GitHub 目标导向**
  - 目标是尽可能将 PDF 压缩到 **100MB 以下**，以适配仓库存储与分发场景。

### 环境要求

- Python 3.12+
- Windows（推荐，DirectML 目标环境）

### 快速开始

安装依赖：

`uv sync`

运行：

`uv run python compress.py`

### 构建 EXE（可选）

`uv run pyinstaller --noconfirm compress.spec`

输出文件：`dist/FireworksPDFCompressor.exe`

### Release 预构建包

- 通过 GitHub Releases 提供预构建 EXE。
- 推送版本标签（如 `v1.0.0`）后自动构建并发布。

### 核心文件

- `compress.py`：程序入口
- `ml_pipeline.py`：ML 流水线调度
- `ml_enhance.py`：模型推理封装
- `adaptive_config.py`：运行配置
- `compress.spec`：EXE 打包配置

### 适用场景

- 仓库中有体积过大的 PDF（如课件、讲义、报告）
- 希望优先满足 GitHub 托管要求，同时尽量保持可读性
- 需要在“压缩率”和“视觉质量”之间平衡

---

## English

A practical **PDF optimization tool for GitHub-hosted documents**:
it aims to shrink large PDFs to **under 100MB** when possible,
while keeping documents readable.

### Objective

- Targeted at GitHub-hosted PDF distribution (100MB constraint).
- Balances compression ratio and readability through a staged pipeline.
- Supports optional ML enhancement and Windows EXE packaging.

### Processing pipeline

- **Stage 1 (lossless-first)**
  1. Structural cleanup
  2. Ghostscript relayout with fallback on text-integrity issues
  3. Safe image recompression
- **Stage 2 (interleaved lossy)**
  - Fixed order: `vector -> image -> tiling`
  - Fixed ladder: `50dB -> 45dB -> 40dB -> 35dB`
  - Compression intensity increases only when needed

### Vector compression

- Vector compression directly reduces the size of drawing-command streams.
- The four strength levels can be interpreted as:
  - **L1 (conservative)**: mild coordinate simplification, usually visually lossless.
  - **L2 (balanced)**: stronger reduction with good size gains on common notes/slides while preserving line continuity.
  - **L3 (aggressive)**: higher compression on complex vector pages, with a small risk of reduced smoothness in dense local details.
  - **L4 (maximum)**: further simplifies tiny curve segments for maximum size reduction; best for distribution scenarios where readability is prioritized over fine geometric fidelity.
- In practice, these levels are applied progressively during the lossy stage instead of jumping directly to the strongest level.

### Compression emphasis by file type

- **Board-writing notes / vector-heavy course materials**: vector + structure compression tends to dominate.
- **Scan/image-heavy PDFs**: image recoding + tiling tends to dominate.
- **Grayscale pages**: text-like regions are binarized aggressively, while gray-detail regions can remain grayscale via hybrid tiling.

- **Codec strategy**
  - Binary data: `FlateDecode`
  - Regular images: `JPXDecode` (JPEG2000)

- **GitHub target**
  - The practical goal is to push PDFs as close as possible to **under 100MB**.

### Requirements

- Python 3.12+
- Windows recommended (DirectML target environment)

### Quick Start

Install dependencies:

`uv sync`

Run:

`uv run python compress.py`

### Build EXE (optional)

`uv run pyinstaller --noconfirm compress.spec`

Output: `dist/FireworksPDFCompressor.exe`

### Prebuilt Releases

- Prebuilt EXE packages are delivered through GitHub Releases.
- Pushing a version tag (e.g. `v1.0.0`) triggers automatic build and release.

### Key Files

- `compress.py`: application entry
- `ml_pipeline.py`: ML pipeline scheduler
- `ml_enhance.py`: model inference wrappers
- `adaptive_config.py`: runtime configuration
- `compress.spec`: EXE build spec

### Best-fit Use Cases

- Large PDFs that are hard to host in GitHub repositories
- Course notes, reports, and scanned documents needing size reduction
- Workflows that require balancing compression ratio and readability

---

## License

Licensed under **MPL-2.0**.

## Third-party Acknowledgements

This project stands on many excellent upstream projects. Thanks to their maintainers and contributors:

- [Ghostscript](https://www.ghostscript.com/)
- [pikepdf](https://github.com/pikepdf/pikepdf) / [qpdf](https://qpdf.sourceforge.io/)
- [PyMuPDF (fitz)](https://github.com/pymupdf/PyMuPDF)
- [ONNX Runtime](https://onnxruntime.ai/) / [onnxruntime-directml](https://pypi.org/project/onnxruntime-directml/)
- [OpenCV](https://opencv.org/), [NumPy](https://numpy.org/), [Pillow](https://python-pillow.org/), [imagecodecs](https://github.com/cgohlke/imagecodecs)
- [doxapy](https://github.com/bacelii/doxapy), [tqdm](https://github.com/tqdm/tqdm), [PyInstaller](https://pyinstaller.org/)

Model ecosystem acknowledgements:

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [NAF-DPM](https://github.com/kuijiang94/NAF-DPM)
- [SLBR (Visible Watermark Removal)](https://github.com/bcmi/SLBR-Visible-Watermark-Removal)

## Distribution & Compliance Notes

For public binary releases, please review these points:

- This repository currently bundles a `gs/` runtime (Ghostscript).
- Ghostscript is commonly distributed under AGPL/commercial dual licensing.
- If you publish prebuilt binaries, include clear source availability and AGPL compliance information in release notes.
- ML model files may have separate redistribution terms; verify each model's license before public redistribution.
