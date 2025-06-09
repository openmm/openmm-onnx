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

#include "internal/OnnxForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <sstream>

using namespace OnnxPlugin;
using namespace OpenMM;
using namespace std;
using namespace Ort;

OnnxForceImpl::OnnxForceImpl(const OnnxForce& owner) : CustomCPPForceImpl(owner), owner(owner), session(nullptr) {
}

OnnxForceImpl::~OnnxForceImpl() {
}

void OnnxForceImpl::initialize(ContextImpl& context) {
    CustomCPPForceImpl::initialize(context);

    // Record which particles the force is applied to.

    particleIndices = owner.getParticleIndices();
    if (particleIndices.size() == 0) {
        int numParticles = context.getSystem().getNumParticles();
        for (int i = 0; i < numParticles; i++)
            particleIndices.push_back(i);
    }

    // Select the execution provider and set options.

    OnnxForce::ExecutionProvider provider = owner.getExecutionProvider();
    string deviceIndex = owner.getProperties().at("DeviceIndex");
    string enableGraph;
    if (owner.getProperties().at("UseGraphs") == "true")
        enableGraph = "1";
    else if (owner.getProperties().at("UseGraphs") == "false")
        enableGraph = "0";
    else
        throw OpenMMException("Illegal value for UseGraphs: "+owner.getProperties().at("UseGraphs"));
    SessionOptions options;
    if (provider == OnnxForce::TensorRT || provider == OnnxForce::Default) {
        OrtTensorRTProviderOptionsV2* rtOptions = nullptr;
        if (GetApi().CreateTensorRTProviderOptions(&rtOptions) == nullptr) {
            vector<const char*> keys{"device_id", "trt_cuda_graph_enable"};
            vector<const char*> values{deviceIndex.c_str(), enableGraph.c_str()};
            ThrowOnError(GetApi().UpdateTensorRTProviderOptions(rtOptions, keys.data(), values.data(), 2));
            options.AppendExecutionProvider_TensorRT_V2(*rtOptions);
        }
        else if (provider == OnnxForce::TensorRT)
            throw OpenMMException("TensorRT execution provider is not available");
    }
    if (provider == OnnxForce::CUDA || provider == OnnxForce::Default) {
        OrtCUDAProviderOptionsV2* cudaOptions = nullptr;
        if (GetApi().CreateCUDAProviderOptions(&cudaOptions) == nullptr) {
            vector<const char*> keys{"device_id", "use_tf32", "enable_cuda_graph"};
            vector<const char*> values{deviceIndex.c_str(), "0", enableGraph.c_str()};
            ThrowOnError(GetApi().UpdateCUDAProviderOptions(cudaOptions, keys.data(), values.data(), 3));
            options.AppendExecutionProvider_CUDA_V2(*cudaOptions);
        }
        else if (provider == OnnxForce::CUDA)
            throw OpenMMException("CUDA execution provider is not available");
    }
    if (provider == OnnxForce::ROCm || provider == OnnxForce::Default) {
        OrtROCMProviderOptions* rocmOptions = nullptr;
        if (GetApi().CreateROCMProviderOptions(&rocmOptions) == nullptr) {
            vector<const char*> keys{"device_id", "enable_hip_graph"};
            vector<const char*> values{deviceIndex.c_str(), enableGraph.c_str()};
            ThrowOnError(GetApi().UpdateROCMProviderOptions(rocmOptions, keys.data(), values.data(), 2));
            options.AppendExecutionProvider_ROCM(*rocmOptions);
        }
        else if (provider == OnnxForce::ROCm)
            throw OpenMMException("ROCm execution provider is not available");
    }

    // Create the session and initialize data structures.

    const vector<uint8_t>& model = owner.getModel();
    session = Session(env, model.data(), model.size(), options);
    positionVec.resize(3*particleIndices.size());
    paramVec.resize(owner.getNumGlobalParameters());
    auto memoryInfo = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    int64_t positionsShape[] = {static_cast<int64_t>(particleIndices.size()), 3};
    int64_t boxShape[] = {3, 3};
    int64_t paramShape[] = {1};
    int numInputs = 1+owner.getNumGlobalParameters();
    if (owner.usesPeriodicBoundaryConditions())
        numInputs++;
    inputTensors.emplace_back(Value::CreateTensor<float>(memoryInfo, positionVec.data(), positionVec.size(), positionsShape, 2));
    inputNames.push_back("positions");
    if (owner.usesPeriodicBoundaryConditions()) {
        inputTensors.emplace_back(Value::CreateTensor<float>(memoryInfo, boxVectors, 9, boxShape, 2));
        inputNames.push_back("box");
    }
    for (int i = 0; i < paramVec.size(); i++) {
        inputTensors.emplace_back(Value::CreateTensor<float>(memoryInfo, &paramVec[i], 1, paramShape, 1));
        inputNames.push_back(owner.getGlobalParameterName(i).c_str());
    }

    // Process extra inputs.

    for (int i = 0; i < owner.getNumInputs(); i++) {
        const OnnxForce::IntegerInput* integerInput = dynamic_cast<const OnnxForce::IntegerInput*>(&owner.getInput(i));
        if (integerInput != nullptr) {
            validateInput(integerInput->getName(), integerInput->getShape(), integerInput->getValues().size());
            integerInputs.push_back(OnnxForce::IntegerInput(integerInput->getName(), integerInput->getValues(), integerInput->getShape()));
        }
        const OnnxForce::FloatInput* floatInput = dynamic_cast<const OnnxForce::FloatInput*>(&owner.getInput(i));
        if (floatInput != nullptr) {
            validateInput(floatInput->getName(), floatInput->getShape(), floatInput->getValues().size());
            floatInputs.push_back(OnnxForce::FloatInput(floatInput->getName(), floatInput->getValues(), floatInput->getShape()));
        }
    }
    for (OnnxForce::IntegerInput& input : integerInputs) {
        vector<int64_t> shape;
        for (int i : input.getShape())
            shape.push_back(i);
        inputTensors.emplace_back(Value::CreateTensor<int>(memoryInfo, input.getValues().data(), input.getValues().size(), shape.data(), shape.size()));
        inputNames.push_back(input.getName().c_str());
    }
    for (OnnxForce::FloatInput& input : floatInputs) {
        vector<int64_t> shape;
        for (int i : input.getShape())
            shape.push_back(i);
        inputTensors.emplace_back(Value::CreateTensor<float>(memoryInfo, input.getValues().data(), input.getValues().size(), shape.data(), shape.size()));
        inputNames.push_back(input.getName().c_str());
    }
}

void OnnxForceImpl::validateInput(const string& name, const vector<int>& shape, int size) {
    int expected = 1;
    for (int i : shape)
        expected *= i;
    if (expected != size) {
        stringstream message;
        message<<"Incorrect length for input '"<<name<<"'.  Expected "<<expected<<" elements, found "<<size<<".";
        throw OpenMMException(message.str());
    }
}

map<string, double> OnnxForceImpl::getDefaultParameters() {
    map<string, double> parameters;
    for (int i = 0; i < owner.getNumGlobalParameters(); i++)
        parameters[owner.getGlobalParameterName(i)] = owner.getGlobalParameterDefaultValue(i);
    return parameters;
}

double OnnxForceImpl::computeForce(ContextImpl& context, const vector<Vec3>& positions, vector<Vec3>& forces) {
    // Pass the current state to ONNX Runtime.

    int numParticles = particleIndices.size();
    for (int i = 0; i < numParticles; i++) {
        int index = particleIndices[i];
        positionVec[3*i] = (float) positions[index][0];
        positionVec[3*i+1] = (float) positions[index][1];
        positionVec[3*i+2] = (float) positions[index][2];
    }
    if (owner.usesPeriodicBoundaryConditions()) {
        Vec3 box[3];
        context.getPeriodicBoxVectors(box[0], box[1], box[2]);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                boxVectors[3*i+j] = (float) box[i][j];
    }
    for (int i = 0; i < owner.getNumGlobalParameters(); i++)
        paramVec[i] = (float) context.getParameter(owner.getGlobalParameterName(i));

    // Perform the computation.

    const char* outputNames[] = {"energy", "forces"};
    outputTensors = session.Run(RunOptions{nullptr}, inputNames.data(), inputTensors.data(), inputNames.size(), outputNames, 2);
    const float* energy = outputTensors[0].GetTensorData<float>();
    const float* forceData = outputTensors[1].GetTensorData<float>();
    for (int i = 0; i < numParticles; i++)
        forces[particleIndices[i]] = Vec3(forceData[3*i], forceData[3*i+1], forceData[3*i+2]);
    return *energy;
}
