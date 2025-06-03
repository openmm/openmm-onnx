/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
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
#include "internal/OnnxForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include <iostream>
#include <fstream>

using namespace OnnxPlugin;
using namespace OpenMM;
using namespace std;

OnnxForce::OnnxForce(const string& file, const map<string, string>& properties) : provider(Default), periodic(false) {
    ifstream input(file, ios::in | ios::binary);
    if (!input.good())
        throw OpenMMException("Failed to read file "+file);
    model = vector<uint8_t>((istreambuf_iterator<char>(input)), istreambuf_iterator<char>());
    initProperties(properties);
}

OnnxForce::OnnxForce(const std::vector<uint8_t>& model, const map<string, string>& properties) : model(model), provider(Default), periodic(false)  {
    initProperties(properties);
}

OnnxForce::~OnnxForce() {
    for (Input* input : inputs)
        delete input;
}

void OnnxForce::initProperties(const std::map<std::string, std::string>& properties) {
    const std::map<std::string, std::string> defaultProperties = {{"UseGraphs", "false"}, {"DeviceIndex", "0"}};
    this->properties = defaultProperties;
    for (auto& property : properties) {
        if (defaultProperties.find(property.first) == defaultProperties.end())
            throw OpenMMException("OnnxForce: Unknown property '" + property.first + "'");
        this->properties[property.first] = property.second;
    }
}

const std::vector<uint8_t>& OnnxForce::getModel() const {
    return model;
}

OnnxForce::ExecutionProvider OnnxForce::getExecutionProvider() const {
    return provider;
}

void OnnxForce::setExecutionProvider(OnnxForce::ExecutionProvider provider) {
    this->provider = provider;
}

const vector<int>& OnnxForce::getParticleIndices() const {
    return particleIndices;
}

void OnnxForce::setParticleIndices(const vector<int>& indices) {
    particleIndices = indices;
}

bool OnnxForce::usesPeriodicBoundaryConditions() const {
    return periodic;
}

void OnnxForce::setUsesPeriodicBoundaryConditions(bool periodic) {
    this->periodic = periodic;
}

ForceImpl* OnnxForce::createImpl() const {
    return new OnnxForceImpl(*this);
}

int OnnxForce::addGlobalParameter(const string& name, double defaultValue) {
    globalParameters.push_back(GlobalParameterInfo(name, defaultValue));
    return globalParameters.size() - 1;
}

int OnnxForce::getNumGlobalParameters() const {
    return globalParameters.size();
}

const string& OnnxForce::getGlobalParameterName(int index) const {
    ASSERT_VALID_INDEX(index, globalParameters);
    return globalParameters[index].name;
}

void OnnxForce::setGlobalParameterName(int index, const string& name) {
    ASSERT_VALID_INDEX(index, globalParameters);
    globalParameters[index].name = name;
}

double OnnxForce::getGlobalParameterDefaultValue(int index) const {
    ASSERT_VALID_INDEX(index, globalParameters);
    return globalParameters[index].defaultValue;
}

void OnnxForce::setGlobalParameterDefaultValue(int index, double defaultValue) {
    ASSERT_VALID_INDEX(index, globalParameters);
    globalParameters[index].defaultValue = defaultValue;
}

int OnnxForce::getNumInputs() const {
    return inputs.size();
}

int OnnxForce::addInput(OnnxForce::Input* input) {
    inputs.push_back(input);
    return inputs.size() - 1;
}

const OnnxForce::Input& OnnxForce::getInput(int index) const {
    ASSERT_VALID_INDEX(index, inputs);
    return *inputs[index];
}

OnnxForce::Input& OnnxForce::getInput(int index) {
    ASSERT_VALID_INDEX(index, inputs);
    return *inputs[index];
}

void OnnxForce::setProperty(const string& name, const string& value) {
    if (properties.find(name) == properties.end())
        throw OpenMMException("OnnxForce: Unknown property '" + name + "'");
    properties[name] = value;
}

const map<string, string>& OnnxForce::getProperties() const {
    return properties;
}
