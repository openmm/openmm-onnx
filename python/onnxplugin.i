%module openmmonnx

%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include <std_map.i>
%include <std_string.i>
%include <std_vector.i>
%include <factory.i>

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

%typemap(out) const std::vector<int>& {
    int len = (*$1).size();
    $result = PyList_New(len);
    for (int i = 0; i < len; i++)
        PyList_SET_ITEM($result, i, PyLong_FromLong((*$1)[i]));
}

%typemap(in) const std::vector<int>& (std::vector<int> values) {
    PyObject* iterator = PyObject_GetIter($input);
    if (iterator == NULL) {
        PyErr_SetString(PyExc_ValueError, "in method $symname, argument $argnum could not be converted to type $type");
        SWIG_fail;
    }
    PyObject* item = NULL;
    while ((item = PyIter_Next(iterator))) {
        int v = (int) PyLong_AsLong(item);
        Py_DECREF(item);
        if (PyErr_Occurred() != NULL) {
            Py_DECREF(iterator);
            PyErr_SetString(PyExc_ValueError, "in method $symname, argument $argnum could not be converted to type $type");
            SWIG_fail;
        }
        values.push_back(v);
    }
    Py_DECREF(iterator);
    $1 = &values;
}

%typecheck(SWIG_TYPECHECK_POINTER) const std::vector<int>& {
    PyObject* iterator = PyObject_GetIter($input);
    $1 = (iterator != NULL);
    Py_DECREF(iterator);
}

%typemap(out) const std::vector<float>& {
    int len = (*$1).size();
    $result = PyList_New(len);
    for (int i = 0; i < len; i++)
        PyList_SET_ITEM($result, i, PyFloat_FromDouble((*$1)[i]));
}

%typemap(in) const std::vector<float>& (std::vector<float> values) {
    PyObject* iterator = PyObject_GetIter($input);
    if (iterator == NULL) {
        PyErr_SetString(PyExc_ValueError, "in method $symname, argument $argnum could not be converted to type $type");
        SWIG_fail;
    }
    PyObject* item = NULL;
    while ((item = PyIter_Next(iterator))) {
        float v = (float) PyFloat_AsDouble(item);
        Py_DECREF(item);
        if (PyErr_Occurred() != NULL) {
            Py_DECREF(iterator);
            PyErr_SetString(PyExc_ValueError, "in method $symname, argument $argnum could not be converted to type $type");
            SWIG_fail;
        }
        values.push_back(v);
    }
    Py_DECREF(iterator);
    $1 = &values;
}

%typecheck(SWIG_TYPECHECK_POINTER) const std::vector<float>& {
    PyObject* iterator = PyObject_GetIter($input);
    $1 = (iterator != NULL);
    Py_DECREF(iterator);
}

%pythonappend OnnxPlugin::OnnxForce::addInput(Input* input) %{
   input.thisown=0
%}

namespace std {
    %template(vectorbyte) vector<unsigned char>;
    %template(property_map) map<std::string, std::string>;
};

%feature("flatnested", "1");

%factory(OnnxPlugin::OnnxForce::Input& OnnxPlugin::OnnxForce::getInput,
         OnnxPlugin::OnnxForce::IntegerInput,
         OnnxPlugin::OnnxForce::FloatInput);

namespace OnnxPlugin {

class OnnxForce : public OpenMM::Force {
public:
    class Input;
    class IntegerInput;
    class FloatInput;
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
    const std::vector<int>& getParticleIndices() const;
    void setParticleIndices(const std::vector<int>& indices);
    bool usesPeriodicBoundaryConditions() const;
    void setUsesPeriodicBoundaryConditions(bool periodic);
    int getNumGlobalParameters() const;
    int addGlobalParameter(const std::string& name, double defaultValue);
    const std::string& getGlobalParameterName(int index) const;
    void setGlobalParameterName(int index, const std::string& name);
    double getGlobalParameterDefaultValue(int index) const;
    void setGlobalParameterDefaultValue(int index, double defaultValue);
    int getNumInputs() const;
    int addInput(Input* input);
    const Input& getInput(int index) const;
    Input& getInput(int index);
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

class OnnxForce::Input {
public:
    const std::string& getName() const;
    const std::vector<int>& getShape() const;
    void setShape(const std::vector<int>& shape);
private:
    Input();
};

class OnnxForce::IntegerInput : public OnnxForce::Input {
public:
    IntegerInput(const std::string& name, const std::vector<int>& values, const std::vector<int>& shape);
    const std::vector<int>& getValues() const;
    void setValues(const std::vector<int>& values);
};

class OnnxForce::FloatInput : public OnnxForce::Input {
public:
    FloatInput(const std::string& name, const std::vector<float>& values, const std::vector<int>& shape);
    const std::vector<float>& getValues() const;
    void setValues(const std::vector<float>& values);
};

}
