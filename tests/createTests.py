import torch

class Central(torch.nn.Module):
    def forward(self, positions):
        positions.grad = None
        energy = torch.sum(positions*positions)
        energy.backward()
        forces = -positions.grad
        return energy, forces

torch.onnx.export(model=Central(),
                  args=(torch.ones(1, 3, requires_grad=True),),
                  f="central.onnx",
                  input_names=["positions"],
                  output_names=["energy", "forces"],
                  dynamic_axes={"positions":[0], "forces":[0]})


class Periodic(torch.nn.Module):
    def forward(self, positions, box):
        positions.grad = None
        boxsize = torch.diagonal(box)
        periodicPositions = positions - torch.floor(positions/boxsize)*boxsize
        energy = torch.sum(periodicPositions*periodicPositions)
        energy.backward()
        forces = -positions.grad
        return energy, forces

torch.onnx.export(model=Periodic(),
                  args=(torch.ones(1, 3, requires_grad=True), torch.ones(3, 3)),
                  f="periodic.onnx",
                  input_names=["positions", "box"],
                  output_names=["energy", "forces"],
                  dynamic_axes={"positions":[0], "forces":[0]})


class Global(torch.nn.Module):
    def forward(self, positions, k):
        positions.grad = None
        energy = k*torch.sum(positions*positions)
        energy.backward()
        forces = -positions.grad
        return energy, forces

torch.onnx.export(model=Global(),
                  args=(torch.ones(1, 3, requires_grad=True), torch.ones(1)),
                  f="global.onnx",
                  input_names=["positions", "k"],
                  output_names=["energy", "forces"],
                  dynamic_axes={"positions":[0], "forces":[0]})


class Inputs(torch.nn.Module):
    def forward(self, positions, scale, offset):
        positions.grad = None
        r = torch.sum(positions*positions, dim=1)
        energy = torch.sum(scale*(r-offset))
        energy.backward()
        forces = -positions.grad
        return energy, forces

torch.onnx.export(model=Inputs(),
                  args=(torch.ones(1, 3, requires_grad=True), torch.ones(1, dtype=torch.int32), torch.ones(1)),
                  f="inputs.onnx",
                  input_names=["positions", "scale", "offset"],
                  output_names=["energy", "forces"],
                  dynamic_axes={"positions":[0], "scale":[0], "offset":[0], "forces":[0]})
