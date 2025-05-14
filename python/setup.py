from setuptools import setup, Extension
import os
import platform

openmm_dir = '@OPENMM_DIR@'
openmmonnx_header_dir = '@OPENMMONNX_HEADER_DIR@'
openmmonnx_library_dir = '@OPENMMONNX_LIBRARY_DIR@'

# setup extra compile and link arguments on Mac
extra_compile_args=['-std=c++14']
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.13']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.13', '-Wl', '-rpath', openmm_dir+'/lib']

extension = Extension(name='_openmmonnx',
                      sources=['OnnxPluginWrapper.cpp'],
                      libraries=['OpenMM', 'OpenMMONNX'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), openmmonnx_header_dir],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), openmmonnx_library_dir],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )

setup(name='OpenMMONNX',
      version='0.1',
      py_modules=['openmmonnx'],
      ext_modules=[extension],
     )
