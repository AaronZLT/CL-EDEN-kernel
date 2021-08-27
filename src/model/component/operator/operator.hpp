#ifndef SRC_MODEL_COMPONENT_OPERATOR_OPERATOR_OPERATOR_HPP_
#define SRC_MODEL_COMPONENT_OPERATOR_OPERATOR_OPERATOR_HPP_

#include <memory>
#include <vector>
#include <string>

#include "model/component/operator/ioperator.hpp"
#include "model/memory/allocated_buffer.hpp"
#include "model/schema/schema_nnc.h"
#include "model/component/operator/data/binary.hpp"
#include "model/component/operator/data/option.hpp"


namespace enn {
namespace model {
namespace component {


// TODO(yc18.cho): Inherit subclasses corresponding to each trait of operators.
//  [Operator] <|-- [GeneralizedOperator] <|-- [ParameterizedOperator]
//  [Operator] <|-- [SpecializedOperator]

//  - data::Option and related data will be migrated to [GeneralizedOperator]
//  - data::Parameter and related data will be migrated to [ParameterizedOperator]
//  - data::Binary and related data will be migrated to [SpecializedOperator]

class Operator : public IOperator {
 public:
    using Ptr = std::shared_ptr<Operator>;

 public:
    virtual ~Operator() = default;

    uint64_t get_id() const {
        return id_;
    }

    const TFlite::BuiltinOperator& get_code() const {
        return code_;
    }

    const std::vector<data::Binary>& get_binaries() const {
        return binaries_;
    }

    const data::Option& get_option() const {
        return option_;
    }

    uint32_t get_in_pixel_format() const {
        return in_pixel_format_;
    }

    bool is_buffer_shared() const {
        return buffer_shared_;
    }

    bool is_ofm_bound() const {
        return ofm_bound_;
    }

    bool is_ifm_bound() const {
        return ifm_bound_;
    }

    bool is_dsp_async_exec() const {
        return dsp_async_exec_;
    }

    const enn::model::Accelerator& get_accelerator() const {
        return accelerator_;
    }

    std::vector<std::string> get_lib_names() const {
        return lib_names_;
    }

 private:
    friend class OperatorBuilder;  // delcare Builder class as friend, which only has creation right.
    // OperatorBuilder only can create Operator object
    Operator() : id_(UNDEFINED), in_pixel_format_(0), buffer_shared_(false), ifm_bound_(false), ofm_bound_(false),
                 dsp_async_exec_(false), code_(TFlite::BuiltinOperator_MIN), option_() {
        binaries_.clear();
        lib_names_.clear();
    }

 private:
    uint64_t                id_;                    // for vertex's ID

    // TODO(yc18.cho): Migrate it to SpecializedOperator derived from this class.
    uint32_t                in_pixel_format_;  // for NCP
    bool                    buffer_shared_;    // for NCP's sharing buffer
    bool                    ifm_bound_;        // for NCP's binding ifm
    bool                    ofm_bound_;        // for NCP's binding ofm
    bool                    dsp_async_exec_;    // for DSP's Async execution with NPU.
    std::vector<data::Binary> binaries_;       // bianry list for operator
    std::vector<std::string> lib_names_;       // for CGO's lib name

    // TODO(yc18.cho): Migrate it to GeneralizedOperator derived from this class.
    TFlite::BuiltinOperator code_;              // enum of BuiltinOperator in schema_generated.h
    data::Option option_;
};


};  // namespace component
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_COMPONENT_OPERATOR_OPERATOR_OPERATOR_HPP_
