from pathlib import Path
from setuptools import setup, Extension
from Cython.Build import cythonize

HERE = Path(__file__).resolve().parent

ext_modules = [
    Extension(
        name="vector_hotspot_cython_nogil",
        sources=[str(HERE / "vector_hotspot_cython_nogil.pyx")],
    ),
]

setup(
    name="vector-hotspot-cython",
    packages=[],
    py_modules=[],
    ext_modules=cythonize(ext_modules, language_level="3", force=True),
)
