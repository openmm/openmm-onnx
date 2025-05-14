%module openmmonnx

%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include <std_map.i>
%include <std_string.i>
%include <std_vector.i>

%{
#include "OnnxForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}

%typemap(out) const std::vector<uint8_t>& {
     $result = PyBytes_FromStringAndSize(reinterpret_cast<const char*>((*$1).data()), (*$1).size());
}

%typemap(in) const std::vector<uint8_t>& (std::vector<uint8_t> model) {
    uint8_t* buffer;
    Py_ssize_t length;
    PyBytes_AsStringAndSize($input, reinterpret_cast<char**>(&buffer), &length);
    model.resize(length);
    for (int i = 0; i < length; i++)
        model[i] = buffer[i];
    $1 = &model;
}

%typecheck(SWIG_TYPECHECK_POINTER) const std::vector<uint8_t>& {
    $1 = PyBytes_Check($input);
}

namespace std {
    %template(vectorbyte) vector<unsigned char>;
    %template(property_map) map<std::string, std::string>;
};

namespace OnnxPlugin {

class OnnxForce : public OpenMM::Force {
public:
    enum ExecutionProvider {
        Default = 0,
        CPU = 1,
        CUDA = 2,
        TensorRT = 3,
        ROCm = 4,
    };
    OnnxForce(const std::string& file, const std::map<std::string, std::string>& properties={});
    OnnxForce(const std::vector<uint8_t>& model, const std::map<std::string, std::string>& properties={});
    const std::vector<uint8_t>& getModel() const;
    ExecutionProvider getExecutionProvider() const;
    void setExecutionProvider(ExecutionProvider provider);
    bool usesPeriodicBoundaryConditions() const;
    void setUsesPeriodicBoundaryConditions(bool periodic);
    int getNumGlobalParameters() const;
    int addGlobalParameter(const std::string& name, double defaultValue);
    const std::string& getGlobalParameterName(int index) const;
    void setGlobalParameterName(int index, const std::string& name);
    double getGlobalParameterDefaultValue(int index) const;
    void setGlobalParameterDefaultValue(int index, double defaultValue);
    void setProperty(const std::string& name, const std::string& value);
    const std::map<std::string, std::string>& getProperties() const;

    /*
     * Add methods for casting a Force to a OnnxForce.
    */
    %extend {
        static OnnxPlugin::OnnxForce& cast(OpenMM::Force& force) {
            return dynamic_cast<OnnxPlugin::OnnxForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<OnnxPlugin::OnnxForce*>(&force) != NULL);
        }
    }
};

}
