import openmm as mm
import openmm.unit as unit
import openmmonnx
import numpy as np
import pytest

def testConstructors():
    model_file = '../../tests/central.onnx'
    force = openmmonnx.OnnxForce(model_file)
    model = open(model_file, 'rb').read()
    assert model == force.getModel()
    force = openmmonnx.OnnxForce(model)
    model = force.getModel()
    force = openmmonnx.OnnxForce(model)

@pytest.mark.parametrize('use_cv_force', [True, False])
@pytest.mark.parametrize('platform', [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())])
def testForce(use_cv_force, platform):

    # Create a random cloud of particles.
    numParticles = 10
    system = mm.System()
    positions = np.random.rand(numParticles, 3)
    for _ in range(numParticles):
        system.addParticle(1.0)

    # Create a force
    force = openmmonnx.OnnxForce('../../tests/central.onnx', {'UseCUDAGraphs': 'false'})
    assert force.getProperties()['UseCUDAGraphs'] == 'false'
    if use_cv_force:
        # Wrap OnnxForce into CustomCVForce
        cv_force = mm.CustomCVForce('force')
        cv_force.addCollectiveVariable('force', force)
        system.addForce(cv_force)
    else:
        system.addForce(force)

    # Compute the forces and energy.
    integ = mm.VerletIntegrator(1.0)
    try:
        context = mm.Context(system, integ, mm.Platform.getPlatformByName(platform))
    except:
        pytest.skip(f'Unable to create Context with {platform}')
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces=True)

    # See if the energy and forces are correct.  The network defines a potential of the form E(r) = |r|^2
    expectedEnergy = np.sum(positions*positions)
    assert np.allclose(expectedEnergy, state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
    assert np.allclose(-2*positions, state.getForces(asNumpy=True))

def testProperties():
    """ Test that the properties are correctly set and retrieved """
    force = openmmonnx.OnnxForce('../../tests/central.onnx')
    force.setProperty('UseCUDAGraphs', 'true')
    assert force.getProperties()['UseCUDAGraphs'] == 'true'
    force.setProperty('UseCUDAGraphs', 'false')
    assert force.getProperties()['UseCUDAGraphs'] == 'false'

def testSerialization():
    force1 = openmmonnx.OnnxForce('../../tests/central.onnx')
    xml1 = mm.XmlSerializer.serialize(force1)
    force2 = mm.XmlSerializer.deserialize(xml1)
    xml2 = mm.XmlSerializer.serialize(force2)
    assert xml1 == xml2