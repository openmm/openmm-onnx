/* -------------------------------------------------------------------------- *
 *                                 OpenMM-NN                                    *
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

#include "OnnxForceProxy.h"
#include "OnnxForce.h"
#include "openmm/serialization/SerializationNode.h"
#include <fstream>
#include <iomanip>
#include <sstream>

using namespace OnnxPlugin;
using namespace OpenMM;
using namespace std;

static string hexEncode(const vector<uint8_t>& input) {
    stringstream ss;
    ss << hex << setfill('0');
    for (const unsigned char& i : input) {
        ss << setw(2) << static_cast<uint64_t>(i);
    }
    return ss.str();
}

static vector<uint8_t> hexDecode(const string& input) {
    vector<uint8_t> res;
    res.reserve(input.size() / 2);
    for (size_t i = 0; i < input.length(); i += 2) {
        istringstream iss(input.substr(i, 2));
        uint64_t temp;
        iss >> hex >> temp;
        res.push_back(static_cast<uint8_t>(temp));
    }
    return res;
}

OnnxForceProxy::OnnxForceProxy() : SerializationProxy("OnnxForce") {
}

void OnnxForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 1);
    const OnnxForce& force = *reinterpret_cast<const OnnxForce*>(object);
    node.setStringProperty("model", hexEncode(force.getModel()));
    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setBoolProperty("usesPeriodic", force.usesPeriodicBoundaryConditions());
    SerializationNode& globalParams = node.createChildNode("GlobalParameters");
    for (int i = 0; i < force.getNumGlobalParameters(); i++)
        globalParams.createChildNode("Parameter").setStringProperty("name", force.getGlobalParameterName(i)).setDoubleProperty("default", force.getGlobalParameterDefaultValue(i));
    SerializationNode& properties = node.createChildNode("Properties");
    for (auto& prop : force.getProperties())
        properties.createChildNode("Property").setStringProperty("name", prop.first).setStringProperty("value", prop.second);
}

void* OnnxForceProxy::deserialize(const SerializationNode& node) const {
    int version = node.getIntProperty("version");
    if (version != 1)
        throw OpenMMException("Unsupported version number");
    OnnxForce* force = new OnnxForce(hexDecode(node.getStringProperty("model")));
    force->setForceGroup(node.getIntProperty("forceGroup"));
    force->setUsesPeriodicBoundaryConditions(node.getBoolProperty("usesPeriodic"));
    for (const SerializationNode& child : node.getChildren()) {
        if (child.getName() == "GlobalParameters")
            for (auto& parameter : child.getChildren())
                force->addGlobalParameter(parameter.getStringProperty("name"), parameter.getDoubleProperty("default"));
        if (child.getName() == "Properties")
            for (auto& property : child.getChildren())
                force->setProperty(property.getStringProperty("name"), property.getStringProperty("value"));
    }
    return force;
}
