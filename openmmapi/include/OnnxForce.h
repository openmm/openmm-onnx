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
#include <vector>

namespace OnnxPlugin {

/**
 * This class uses ONNX Runtime to compute forces defined by a neural network in ONNX format.
 */

class OPENMM_EXPORT_ONNX OnnxForce : public OpenMM::Force {
public:
    /**
     * Create an OnnxForce by loading the ONNX model from a file.
     *
     * @param file       the path to the file containing the model
     */
    OnnxForce(const std::string& file);
    /**
     * Create an OnnxForce by loading the ONNX model from a vector.
     *
     * @param model      the binary representation of the model in ONNX format
     */
    OnnxForce(const std::vector<uint8_t>& model);
    /**
     * Get the binary representation of the model in ONNX format.
     */
    const std::vector<uint8_t>& getModel() const;
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
protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    class GlobalParameterInfo;
    std::vector<uint8_t> model;
    bool periodic;
    std::vector<GlobalParameterInfo> globalParameters;
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
