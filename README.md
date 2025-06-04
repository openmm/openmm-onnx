# OpenMM-ONNX

This is a plugin for [OpenMM](https://openmm.org) that allows forces to be defined by neural networks
saved in [ONNX](https://onnx.ai) format.  This is a cross platform format that can be created with a
variety of machine learning frameworks, such as [PyTorch](https://pytorch.org) or [JAX](https://jax.dev).

## Installation

OpenMM-ONNX currently needs to be compiled from source.  First you need to install the tools it depends
on: OpenMM, ONNX Runtime, CMake, and SWIG.  This is most easily done with conda or mamba:

```
mamba install -c conda-forge openmm onnxruntime onnxruntime-cpp cmake swig
```

To compile, follow these steps.

1. Create a directory in which to build the plugin.

2. Run the CMake GUI or `ccmake`, specifying your new directory as the build directory and the top
level directory of this project as the source directory.

3. Press "Configure".

4. Usually CMake will be able to locate all the dependencies on its own.  Check for any variables that are 
listed as `NOTFOUND` and set them yourself.  `OPENMM_DIR` should point to the top level directory where OpenMM
is installed.  `ONNX_HEADER_PATH` should point to the file `onnxruntime_cxx_api.h`.  `ONNX_LIBRARY_PATH`
should point to the `onnxruntime` library.  (Its exact name will depend on your operating system, for example
`libonnxruntime.so` on Linux.)

5. Press "Configure" again if necessary, then press "Generate".

6. Use the build system you selected to build and install the plugin.  For example, if you
selected Unix Makefiles, type `make install` to install the plugin, and `make PythonInstall` to
install the Python wrapper.

## Creating a Model

Now you need to create the ONNX model.  In the simplest case it will have a single input of shape
`(# particles, 3)` called `positions`.  This will contain the particle coordinates in nanometers.
It should produce two outputs: a scalar called `energy` with the potential energy in kJ/mol, and
a tensor of shape `(# particles, 3)` called `forces` with the forces acting on the particles in
kJ/mol/nm.  All inputs and outputs should be 32 bit floating point values.

The following example uses PyTorch to create a simple model that attracts every particle to the
origin with a potential of the form $E(x) = x^2$.  It computes the energy, then uses backpropagation to
compute the forces.  Finally it exports the model to a file called `MyForce.onnx`.

```python
import torch

class MyForce(torch.nn.Module):
    def forward(self, positions):
        positions.grad = None
        energy = torch.sum(positions*positions)
        energy.backward()
        forces = -positions.grad
        return energy, forces

torch.onnx.export(model=MyForce(),
                  args=(torch.ones(1, 3, requires_grad=True),),
                  f="MyForce.onnx",
                  input_names=["positions"],
                  output_names=["energy", "forces"],
                  dynamic_axes={"positions":[0], "forces":[0]})
```

`torch.onnx.export()` executes the model with a set of example arguments and records a trace
of all operations that get performed.  Notice how we use `dynamic_axes` to specify that the first
axis of the `positions` input and `forces` output are dynamic.  This allows the model to be used
for simulating any number of particles.

Using the model in OpenMM is very easy.  Create an `OnnxForce` object, providing the name of the
file you saved the model to.  Add the force to your system, and it works exactly like any other
force.

```python
from openmmonnx import OnnxForce
force = OnnxForce("MyForce.onnx")
system.addForce(force)
```

## Periodic Boundary Conditions

If you want your simulation to use periodic boundary conditions, call `setUsesPeriodicBoundaryConditions(True)`
on the `OnnxForce`.  If you do that, it will expect the model to have a second input called `box`.  It should
be a tensor of shape `(3, 3)` containing the periodic box vectors.  The following example creates a model that
respects periodic boundary conditions.

```python
class PeriodicForce(torch.nn.Module):
    def forward(self, positions, box):
        positions.grad = None
        boxsize = torch.diagonal(box)
        # The following line is only correct for rectangular boxes.
        # Triclinic boxes need a slightly more complicated calculation.
        periodicPositions = positions - torch.floor(positions/boxsize)*boxsize
        energy = torch.sum(periodicPositions*periodicPositions)
        energy.backward()
        forces = -positions.grad
        return energy, forces

torch.onnx.export(model=PeriodicForce(),
                  args=(torch.ones(1, 3, requires_grad=True), torch.ones(3, 3)),
                  f="PeriodicForce.onnx",
                  input_names=["positions", "box"],
                  output_names=["energy", "forces"],
                  dynamic_axes={"positions":[0], "forces":[0]})
```

## Applying to a Subset of Particles

In some cases one wants to model part of a system with a machine learning potential and the rest with a
conventional force field.  You can restrict which particles the `OnnxForce` acts on by calling `setParticleIndices()`.
For example, the following applies it only to the first 50 particles in the system.

```python
particles = list(range(50))
force.setParticleIndices(particles)
```

The `positions` tensor passed to the model will contain only the positions of the specified particles.
Likewise, the `forces` tensor returned by the model should contain only the forces on those particles.

## Global Parameters

An `OnnxForce` can define global parameters that the model depends on.  The model should have an additional
scalar input for each parameter.  For example, this model has a force constant `k` that is defined as a
parameter.

```python
class ForceWithParameter(torch.nn.Module):
    def forward(self, positions, k):
        positions.grad = None
        energy = k*torch.sum(positions*positions)
        energy.backward()
        forces = -positions.grad
        return energy, forces

torch.onnx.export(model=ForceWithParameter(),
                  args=(torch.ones(1, 3, requires_grad=True), torch.ones(1)),
                  f="ForceWithParameter.onnx",
                  input_names=["positions", "k"],
                  output_names=["energy", "forces"],
                  dynamic_axes={"positions":[0], "forces":[0]})
```

Call `addGlobalParameter()` on the `OnnxForce` to define the parameter and set its default value.

```python
force = OnnxForce("ForceWithParameter.onnx")
force.addGlobalParameter("k", 1.0)
```

You can change the parameter value at any time during a simulation by calling `setParameter()`
on the `Context`.

```python
context.setParameter("k", 5.0)
```

## Extra Inputs

You also can specify extra inputs that should be passed to the model.  Unlike global parameters,
which are always scalars, extra inputs can be tensors of any size and shape.  On the other hand,
their values are fixed at Context creation time, and can only be changed by reinitializing the
Context.

This example is similar to the one above, but `k` is now a vector containing a different force
constant for every particle.

```python
class ForceWithInput(torch.nn.Module):
    def forward(self, positions, k):
        positions.grad = None
        r2 = torch.sum(positions*positions, dim=1)
        energy = torch.sum(k*r2)
        energy.backward()
        forces = -positions.grad
        return energy, forces

torch.onnx.export(model=ForceWithInput(),
                  args=(torch.ones(1, 3, requires_grad=True), torch.ones(1)),
                  f="ForceWithInput.onnx",
                  input_names=["positions", "k"],
                  output_names=["energy", "forces"],
                  dynamic_axes={"positions":[0], "forces":[0], "k":[0]})
```

Notice that we included `k[0]` in `dynamic_axes` when exporting the model.  This allows its length
to be variable, so we can use the model for systems with any number of particles.

Here is how we create the OnnxForce.

```python
import openmmonnx
force = OnnxForce("ForceWithInput.onnx")
force.addInput(openmmonnx.FloatInput("k", k, [len(k)]))
```

The three arguments to the FloatInput constructor are the name of the input (matching the name we
specified in `input_names` when exporting the model), a list or array containing the values, and
the shape of the tensor.  In this case the tensor has one dimension, so the shape argument contains
only a single value.  Higher dimensional tensors are also allowed.  In that case, the second
argument should contain the values in flattened order.

In addition to FloatInput, which specifies a tensor of 32 bit floating point values, there is also
an IntegerInput class, which specifies a tensor of 32 bit integer values.

## Execution Providers

ONNX Runtime supports a variety of backends that can be used to compute the neural network.  They
are called "execution providers".  This allows a model to run efficiently on different types of
hardware.

ONNX execution providers are similar to OpenMM platforms, but they are independent of it.  For example,
you can run a simulation with OpenMM's CPU platform, but still have it compute the neural network on
a GPU by using the CUDA or ROCm execution provider.

Most execution providers are not bundled with ONNX Runtime.  They must be installed separately.
See [the documentation](https://onnxruntime.ai/docs/execution-providers/) for details on how to
install particular providers.  You can get a list of what providers are installed with the following
Python commands.

```python
import onnxruntime
print(onnxruntime.get_available_providers())
```

By default, `OnnxForce` will automatically select a provider from among the ones you have installed.
Usually its default choice will be the fastest available provider.  You can override its choice and
force it to use a particular provider by calling `setExecutionProvider()`.  For example, the following
line forces it to evaluate the model on the CPU.

```python
force.setExecutionProvider(OnnxForce.CPU)
```

If you specify a provider that is not available, it will throw an exception when you try to create
a context.

## Properties

Properties are a mechanism for customizing how the calculation is done by particular execution
providers.  You set their values by calling `setProperty()` on an `OnnxForce`.  For example, to
tell it which GPU to use you might call

```python
force.setProperty("DeviceIndex", "1")
```

The following properties are currently supported.

- `"DeviceIndex"`: the index of the GPU to use.  This affects the CUDA, ROCm, and TensorRT providers.
- `"UseGraphs"`: set to `"true"` or `"false"` to specify whether to use CUDA/HIP graphs to optimize
  the calculation.  This can improve performance in some cases, but may not be compatible with all
  models.  It affects the CUDA, ROCm, and TensorRT providers