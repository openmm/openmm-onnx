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
    int numParticles = context.getSystem().getNumParticles();
    positionVec.resize(3*numParticles);
    paramVec.resize(owner.getNumGlobalParameters());
    auto memoryInfo = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    int64_t positionsShape[] = {numParticles, 3};
    int64_t boxShape[] = {3, 3};
    int64_t paramShape[] = {1};
    int numInputs = 1+owner.getNumGlobalParameters();
    if (owner.usesPeriodicBoundaryConditions())
        numInputs++;
    inputTensors.resize(numInputs);
    inputNames.resize(numInputs);
    inputTensors[0] = Value::CreateTensor<float>(memoryInfo, positionVec.data(), positionVec.size(), positionsShape, 2);
    inputNames[0] = "positions";
    if (owner.usesPeriodicBoundaryConditions()) {
        inputTensors[1] = Value::CreateTensor<float>(memoryInfo, boxVectors, 9, boxShape, 2);
        inputNames[1] = "box";
    }
    int paramStart = (owner.usesPeriodicBoundaryConditions() ? 2 : 1);
    for (int i = 0; i < paramVec.size(); i++) {
        inputTensors[i+paramStart] = Value::CreateTensor<float>(memoryInfo, &paramVec[i], 1, paramShape, 1);
        inputNames[i+paramStart] = owner.getGlobalParameterName(i).c_str();
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

    int numParticles = context.getSystem().getNumParticles();
    for (int i = 0; i < numParticles; i++) {
        positionVec[3*i] = positions[i][0];
        positionVec[3*i+1] = positions[i][1];
        positionVec[3*i+2] = positions[i][2];
    }
    if (owner.usesPeriodicBoundaryConditions()) {
        Vec3 box[3];
        context.getPeriodicBoxVectors(box[0], box[1], box[2]);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                boxVectors[3*i+j] = box[i][j];
    }
    for (int i = 0; i < owner.getNumGlobalParameters(); i++)
        paramVec[i] = context.getParameter(owner.getGlobalParameterName(i));

    // Perform the computation.

    const char* outputNames[] = {"energy", "forces"};
    outputTensors = session.Run(RunOptions{nullptr}, inputNames.data(), inputTensors.data(), inputNames.size(), outputNames, 2);
    const float* energy = outputTensors[0].GetTensorData<float>();
    const float* forceData = outputTensors[1].GetTensorData<float>();
    for (int i = 0; i < numParticles; i++)
        forces[i] = Vec3(forceData[3*i], forceData[3*i+1], forceData[3*i+2]);
    return *energy;
}
