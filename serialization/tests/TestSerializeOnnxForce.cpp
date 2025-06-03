/* -------------------------------------------------------------------------- *
 *                                  OpenMM                                    *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2025 Stanford University and the Authors.           *
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
#include "openmm/Platform.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/serialization/XmlSerializer.h"
#include <iostream>
#include <sstream>

using namespace OnnxPlugin;
using namespace OpenMM;
using namespace std;

extern "C" void registerOnnxSerializationProxies();

void testSerialization() {
    OnnxForce force("tests/central.onnx");
    force.setForceGroup(3);
    force.addGlobalParameter("x", 1.3);
    force.addGlobalParameter("y", 2.221);
    force.setUsesPeriodicBoundaryConditions(true);
    force.setProperty("UseGraphs", "true");
    force.setParticleIndices({0, 2, 4});
    force.addInput(new OnnxForce::IntegerInput("ints", {0, 1, 2, 3, 4, 5}, {2, 3}));
    force.addInput(new OnnxForce::FloatInput("floats", {2.0, 4.5, 5.3}, {1, 3}));

    // Serialize and then deserialize it.

    stringstream buffer;
    XmlSerializer::serialize<OnnxForce>(&force, "Force", buffer);
    OnnxForce* copy = XmlSerializer::deserialize<OnnxForce>(buffer);

    // Compare the two forces to see if they are identical.

    OnnxForce& force2 = *copy;
    ASSERT_EQUAL_CONTAINERS(force.getModel(), force2.getModel());
    ASSERT_EQUAL(force.getForceGroup(), force2.getForceGroup());
    ASSERT_EQUAL_CONTAINERS(force.getParticleIndices(), force2.getParticleIndices());
    ASSERT_EQUAL(force.getNumInputs(), force2.getNumInputs());
    for (int i = 0; i < force.getNumInputs(); i++) {
        ASSERT_EQUAL(force.getInput(i).getName(), force2.getInput(i).getName());
        ASSERT_EQUAL_CONTAINERS(force.getInput(i).getShape(), force2.getInput(i).getShape());
    }
    ASSERT_EQUAL_CONTAINERS(dynamic_cast<OnnxForce::IntegerInput&>(force.getInput(0)).getValues(), dynamic_cast<OnnxForce::IntegerInput&>(force2.getInput(0)).getValues());
    ASSERT_EQUAL_CONTAINERS(dynamic_cast<OnnxForce::FloatInput&>(force.getInput(1)).getValues(), dynamic_cast<OnnxForce::FloatInput&>(force2.getInput(1)).getValues());
    ASSERT_EQUAL(force.getNumGlobalParameters(), force2.getNumGlobalParameters());
    for (int i = 0; i < force.getNumGlobalParameters(); i++) {
        ASSERT_EQUAL(force.getGlobalParameterName(i), force2.getGlobalParameterName(i));
        ASSERT_EQUAL(force.getGlobalParameterDefaultValue(i), force2.getGlobalParameterDefaultValue(i));
    }
    ASSERT_EQUAL(force.usesPeriodicBoundaryConditions(), force2.usesPeriodicBoundaryConditions());
    ASSERT_EQUAL(force.getProperties().size(), force2.getProperties().size());
    for (auto& prop : force.getProperties())
        ASSERT_EQUAL(prop.second, force2.getProperties().at(prop.first));
}

int main() {
    try {
        registerOnnxSerializationProxies();
        testSerialization();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
