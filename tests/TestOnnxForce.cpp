/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2018-2025 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "OnnxForce.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "sfmt/SFMT.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace OnnxPlugin;
using namespace OpenMM;
using namespace std;

void testForce(Platform& platform, vector<int> particleIndices) {
    // Create a random cloud of particles.

    const int numParticles = 10;
    System system;
    vector<Vec3> positions(numParticles);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    for (int i = 0; i < numParticles; i++) {
        system.addParticle(1.0);
        positions[i] = Vec3(genrand_real2(sfmt), genrand_real2(sfmt), genrand_real2(sfmt))*10;
    }
    OnnxForce* force = new OnnxForce("tests/central.onnx");
    force->setParticleIndices(particleIndices);
    if (particleIndices.size() == 0)
        for (int i = 0; i < numParticles; i++)
            particleIndices.push_back(i);
    system.addForce(force);

    // Compute the forces and energy.

    VerletIntegrator integ(1.0);
    Context context(system, integ, platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);

    // See if the energy is correct.  The network defines a potential of the form E(r) = |r|^2

    double expectedEnergy = 0;
    for (int i = 0; i < numParticles; i++) {
        if (find(particleIndices.begin(), particleIndices.end(), i) != particleIndices.end()) {
            Vec3 pos = positions[i];
            double r = sqrt(pos.dot(pos));
            expectedEnergy += r*r;
            ASSERT_EQUAL_VEC(pos*(-2.0), state.getForces()[i], 1e-5);
        }
        else {
            Vec3 zero;
            ASSERT_EQUAL_VEC(zero, state.getForces()[i], 1e-5);
        }
    }
    ASSERT_EQUAL_TOL(expectedEnergy, state.getPotentialEnergy(), 1e-5);
}

void testPeriodicForce(Platform& platform) {
    // Create a random cloud of particles.

    const int numParticles = 10;
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(2, 0, 0), Vec3(0, 3, 0), Vec3(0, 0, 4));
    vector<Vec3> positions(numParticles);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    for (int i = 0; i < numParticles; i++) {
        system.addParticle(1.0);
        positions[i] = Vec3(genrand_real2(sfmt), genrand_real2(sfmt), genrand_real2(sfmt))*10;
    }
    OnnxForce* force = new OnnxForce("tests/periodic.onnx");
    force->setUsesPeriodicBoundaryConditions(true);
    system.addForce(force);

    // Compute the forces and energy.

    VerletIntegrator integ(1.0);
    Context context(system, integ, platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);

    // See if the energy is correct.  The network defines a potential of the form E(r) = |r|^2

    double expectedEnergy = 0;
    for (int i = 0; i < numParticles; i++) {
        Vec3 pos = positions[i];
        pos[0] -= floor(pos[0]/2.0)*2.0;
        pos[1] -= floor(pos[1]/3.0)*3.0;
        pos[2] -= floor(pos[2]/4.0)*4.0;
        double r = sqrt(pos.dot(pos));
        expectedEnergy += r*r;
        ASSERT_EQUAL_VEC(pos*(-2.0), state.getForces()[i], 1e-5);
    }
    ASSERT_EQUAL_TOL(expectedEnergy, state.getPotentialEnergy(), 1e-5);
}

void testGlobal(Platform& platform) {
    // Create a random cloud of particles.

    const int numParticles = 10;
    System system;
    vector<Vec3> positions(numParticles);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    for (int i = 0; i < numParticles; i++) {
        system.addParticle(1.0);
        positions[i] = Vec3(genrand_real2(sfmt), genrand_real2(sfmt), genrand_real2(sfmt))*10;
    }
    OnnxForce* force = new OnnxForce("tests/global.onnx");
    force->addGlobalParameter("k", 2.0);
    system.addForce(force);

    // Compute the forces and energy.

    VerletIntegrator integ(1.0);
    Context context(system, integ, platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);

    // See if the energy is correct.  The network defines a potential of the form E(r) = k*|r|^2

    double expectedEnergy = 0;
    for (int i = 0; i < numParticles; i++) {
        Vec3 pos = positions[i];
        double r = sqrt(pos.dot(pos));
        expectedEnergy += 2*r*r;
        ASSERT_EQUAL_VEC(pos*(-4.0), state.getForces()[i], 1e-5);
    }
    ASSERT_EQUAL_TOL(expectedEnergy, state.getPotentialEnergy(), 1e-5);

    // Change the global parameter and see if the forces are still correct.

    context.setParameter("k", 3.0);
    state = context.getState(State::Forces | State::Energy);
    for (int i = 0; i < numParticles; i++) {
        Vec3 pos = positions[i];
        double r = sqrt(pos.dot(pos));
        ASSERT_EQUAL_VEC(pos*(-6.0), state.getForces()[i], 1e-5);
    }
    ASSERT_EQUAL_TOL(expectedEnergy*1.5, state.getPotentialEnergy(), 1e-5);
}

void testInputs(Platform& platform) {
    // Create a random cloud of particles.

    const int numParticles = 10;
    System system;
    vector<Vec3> positions(numParticles);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    for (int i = 0; i < numParticles; i++) {
        system.addParticle(1.0);
        positions[i] = Vec3(genrand_real2(sfmt), genrand_real2(sfmt), genrand_real2(sfmt))*10;
    }
    OnnxForce* force = new OnnxForce("tests/inputs.onnx");
    system.addForce(force);

    // Define extra inputs.

    vector<int> scale = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    vector<float> offset = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    force->addInput(new OnnxForce::IntegerInput("scale", scale, {10}));
    force->addInput(new OnnxForce::FloatInput("offset", offset, {10}));

    // Compute the forces and energy.

    VerletIntegrator integ(1.0);
    Context context(system, integ, platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);

    // See if the energy is correct.  The network defines a potential of the form E(r) = scale*(|r|^2-offset).

    double expectedEnergy = 0;
    for (int i = 0; i < numParticles; i++) {
        Vec3 pos = positions[i];
        double r = sqrt(pos.dot(pos));
        expectedEnergy += scale[i]*(r*r-offset[i]);
        ASSERT_EQUAL_VEC(pos*(-2.0*scale[i]), state.getForces()[i], 1e-5);
    }
    ASSERT_EQUAL_TOL(expectedEnergy, state.getPotentialEnergy(), 1e-5);
}

void testPlatform(Platform& platform) {
    testForce(platform, {});
    testForce(platform, {0, 1, 2, 9, 5});
    testPeriodicForce(platform);
    testGlobal(platform);
    testInputs(platform);
}

int main(int argc, char* argv[]) {
    try {
        Platform::loadPluginsFromDirectory(Platform::getDefaultPluginsDirectory());
        for (int i = 0; i < Platform::getNumPlatforms(); i++) {
            Platform& platform = Platform::getPlatform(i);
            printf("Testing %s\n", platform.getName().c_str());
            testPlatform(platform);
        }
    }
    catch(const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
