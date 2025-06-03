#ifndef OPENMM_ONNXFORCE_H_
#define OPENMM_ONNXFORCE_H_

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

#include "openmm/Context.h"
#include "openmm/Force.h"
#include "internal/windowsExportOnnx.h"
#include <map>
#include <vector>

namespace OnnxPlugin {

/**
 * This class uses ONNX Runtime to compute forces defined by a neural network in ONNX format.
 */

class OPENMM_EXPORT_ONNX OnnxForce : public OpenMM::Force {
public:
    class Input;
    class IntegerInput;
    class FloatInput;
    class BoolInput;
    /**
     * This is an enumeration of ONNX execution providers.
     */
    enum ExecutionProvider {
        /**
         * Select an execution provider automatically, based on which ones are available.
         * Usually this will select the fastest available provider.
         */
        Default = 0,
        /**
         * Compute the model on the CPU.  This is the only provider that is guaranteed to
         * always be available.
         */
        CPU = 1,
        /**
         * Use the CUDA execution provider.  This is only available on NVIDIA GPUs.
         */
        CUDA = 2,
        /**
         * Use the TensorRT execution provider.  This is only available on NVIDIA GPUs,
         * and requires TensorRT to be installed.
         */
        TensorRT = 3,
        /**
         * Use the ROCm execution provider.  This is most often used for AMD GPUs, but
         * may sometimes be available for other hardware as well.
         */
        ROCm = 4,
    };
    /**
     * Create an OnnxForce by loading the ONNX model from a file.
     *
     * @param file       the path to the file containing the model
     * @param properties optional map of properties
     */
    OnnxForce(const std::string& file, const std::map<std::string, std::string>& properties={});
    /**
     * Create an OnnxForce by loading the ONNX model from a vector.
     *
     * @param model      the binary representation of the model in ONNX format
     * @param properties optional map of properties
     */
    OnnxForce(const std::vector<uint8_t>& model, const std::map<std::string, std::string>& properties={});
    ~OnnxForce();
    /**
     * Get the binary representation of the model in ONNX format.
     */
    const std::vector<uint8_t>& getModel() const;
    /**
     * Get the execution provider to be used for computing the model.
     */
    ExecutionProvider getExecutionProvider() const;
    /**
     * Set the execution provider to be used for computing the model.
     */
    void setExecutionProvider(ExecutionProvider provider);
    /**
     * Get the indices of the particles this force is applied to.  If this is empty, the
     * force is applied to all particles in the system.
     */
    const std::vector<int>& getParticleIndices() const;
    /**
     * Set the indices of the particles this force is applied to.  If this is empty, the
     * force is applied to all particles in the system.
     */
    void setParticleIndices(const std::vector<int>& indices);
    /**
     * Get whether this force uses periodic boundary conditions.
     */
    bool usesPeriodicBoundaryConditions() const;
    /**
     * Set whether this force uses periodic boundary conditions.
     */
    void setUsesPeriodicBoundaryConditions(bool periodic);
    /**
     * Get the number of global parameters that the interaction depends on.
     */
    int getNumGlobalParameters() const;
    /**
     * Add a new global parameter that the interaction may depend on.  The default value provided to
     * this method is the initial value of the parameter in newly created Contexts.  You can change
     * the value at any time by calling setParameter() on the Context.
     *
     * @param name             the name of the parameter
     * @param defaultValue     the default value of the parameter
     * @return the index of the parameter that was added
     */
    int addGlobalParameter(const std::string& name, double defaultValue);
    /**
     * Get the name of a global parameter.
     *
     * @param index     the index of the parameter for which to get the name
     * @return the parameter name
     */
    const std::string& getGlobalParameterName(int index) const;
    /**
     * Set the name of a global parameter.
     *
     * @param index          the index of the parameter for which to set the name
     * @param name           the name of the parameter
     */
    void setGlobalParameterName(int index, const std::string& name);
    /**
     * Get the default value of a global parameter.
     *
     * @param index     the index of the parameter for which to get the default value
     * @return the parameter default value
     */
    double getGlobalParameterDefaultValue(int index) const;
    /**
     * Set the default value of a global parameter.
     *
     * @param index          the index of the parameter for which to set the default value
     * @param defaultValue   the default value of the parameter
     */
    void setGlobalParameterDefaultValue(int index, double defaultValue);
    /**
     * Get the number of extra tensors to pass to the model.
     */
    int getNumInputs() const;
    /**
     * Add an extra tensor that should be passed to the model.  The Input object should have
     * been created on the heap with the "new" operator.  The OnnxForce takes over ownership
     * of it, and deletes it when the OnnxForce itself is deleted.
     *
     * @param input       the tensor to pass in.  This should be the appropriate subclass for
     *                    the type of value.
     * @return the index of the input that was added
     */
    int addInput(Input* input);
    /**
     * Get a const reference to an extra input that is passed to the model.
     *
     * @param index     the index of the input to return
     */
    const Input& getInput(int index) const;
    /**
     * Get a writable reference to an extra input that is passed to the model.
     *
     * @param index     the index of the input to return
     */
    Input& getInput(int index);
    /**
     * Set the value of a property.
     *
     * @param name           the name of the property
     * @param value          the value of the property
     */
    void setProperty(const std::string& name, const std::string& value);
    /**
     * Get the map of properties for this instance.
     * @return A map of property names to values.
     */
    const std::map<std::string, std::string>& getProperties() const;
protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    class GlobalParameterInfo;
    void initProperties(const std::map<std::string, std::string>& properties);
    std::vector<uint8_t> model;
    std::vector<int> particleIndices;
    ExecutionProvider provider;
    bool periodic;
    std::vector<GlobalParameterInfo> globalParameters;
    std::vector<Input*> inputs;
    std::map<std::string, std::string> properties;
};

/**
 * An Input defines a tensor that should be passed to the model.  This is an abstract class.  Subclasses
 * define particular types of tensors.
 */
class OnnxForce::Input {
public:
    virtual ~Input() {
    }
    /**
     * Get the name of the input.
     */
    const std::string& getName() const {
        return name;
    }
    /**
     * Get the shape of the tensor.
     */
    const std::vector<int>& getShape() const {
        return shape;
    }
    /**
     * Set the shape of the tensor.
     */
    void setShape(const std::vector<int>& shape) {
        this->shape = shape;
    }
protected:
    Input() {
    }
    Input(const std::string& name, const std::vector<int>& shape) : name(name), shape(shape) {
    }
private:
    std::string name;
    std::vector<int> shape;
};

/**
 * A tensor containing integer values that should be passed to the model.
 */
class OnnxForce::IntegerInput : public Input {
public:
    /**
     * Create an IntegerInput.
     *
     * @param name      the name of the input
     * @param values    the values contained by the tensor, in flattened order
     * @param shape     the shape of the tensor
     */
    IntegerInput(const std::string& name, const std::vector<int>& values, const std::vector<int>& shape) : Input(name, shape), values(values) {
    }
    /**
     * Get a const reference to the values contained in the tensor, in flattened order.
     */
    const std::vector<int>& getValues() const {
        return values;
    }
    /**
     * Get a writable reference to the values contained in the tensor, in flattened order.
     */
    std::vector<int>& getValues() {
        return values;
    }
    /**
     * Set the values contained in the tensor, in flattened order.
     */
    void setValues(const std::vector<int>& values) {
        this->values = values;
    }
private:
    std::vector<int> values;
};

/**
 * A tensor containing float values that should be passed to the model.
 */
class OnnxForce::FloatInput : public Input {
public:
    /**
     * Create an FloatInput.
     *
     * @param name      the name of the input
     * @param values    the values contained by the tensor, in flattened order
     * @param shape     the shape of the tensor
     */
    FloatInput(const std::string& name, const std::vector<float>& values, const std::vector<int>& shape) : Input(name, shape), values(values) {
    }
    /**
     * Get a const reference to the values contained in the tensor, in flattened order.
     */
    const std::vector<float>& getValues() const {
        return values;
    }
    /**
     * Get a writable reference to the values contained in the tensor, in flattened order.
     */
    std::vector<float>& getValues() {
        return values;
    }
    /**
     * Set the values contained in the tensor, in flattened order.
     */
    void setValues(const std::vector<float>& values) {
        this->values = values;
    }
private:
    std::vector<float> values;
};

/**
 * This is an internal class used to record information about a global parameter.
 * @private
 */
class OnnxForce::GlobalParameterInfo {
public:
    std::string name;
    double defaultValue;
    GlobalParameterInfo() {
    }
    GlobalParameterInfo(const std::string& name, double defaultValue) : name(name), defaultValue(defaultValue) {
    }
};

} // namespace OnnxPlugin

#endif /*OPENMM_ONNXFORCE_H_*/
