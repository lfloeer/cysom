from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

SOM_MODULE = Extension(
    "cysom.som",
    ["cysom/som.pyx"],
    extra_compile_args=['-O3'],
    include_dirs=[numpy.get_include()]
)

setup(
    name='CySOM',
    version='1.0',
    description='Simple but fast self-organizing map implementation in Cython.',
    author='Lars Flöer',
    author_email='mail@lfloeer.de',
    cmdclass={'build_ext': build_ext},
    ext_modules=[SOM_MODULE],
    packages=['cysom']
)
