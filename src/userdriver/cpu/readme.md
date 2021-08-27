# Userdriver(UD) Interface

## CPU UD Interface

- CPU UD initializes and executes CPU operators requested by ENN engine.
- CPU UD manages the map of UDOperators created by OperatorConstructor.
- OperatorConstructor collects the options or parameters of each operator and then initializes the operator with it.
- OperatorExecutor set the input/output buffer address allocated from user(client layer) to each operator and then executes the operator.
- NEONComputeLibrary creates each CPU operator's object pointer. And supports methods to create tensor that it is a data tranfered to each operator.


```plantuml
@startuml
scale 800 width
package "enn" #DDDDDD {
    package "ud" #DDDDDD {
        class UserDriver {
            +Initialize()
            +OpenSubGraph()
            +PrepareSubGraph()
            +ExecuteSubGraph()
            +CloseSubGraph()
            +Deinitialize()
        }

        class IOperationConstructor {
            +initialize_ud_operators()
            +create_ud_operator()
            +get_ud_operators()
        }

        class IOperationExecutor {
            +run()
        }

        class IComputeLibrary {
            +assignBuffers()
            +flush()
            +synchronize()
            +create_tensor()
            +create_and_copy_tensor()
            +createNormalization()
            +createDequantization()
            +...()
        }

        package "cpu" #DDDDDD {
            class CpuUserDriver {
                +get_instance()
                +Initialize()
                +OpenSubGraph()
                +ExecuteSubGraph()
                +CloseSubGraph()
                +Deinitialize()
                -compute_library
                -op_constructor
                -op_executor
            }

            class OperationConstructor {
                +initialize_ud_operators()
                +create_ud_operator()
                +get_ud_operators()
                +create_ud_operator<T>()
                -is_builtin_operator()
                -convert_to_tensors()
                -create_and_copy_tensor()
                -builtin_op_map
                -custom_op_map
                -compute_library
                -operators
            }

            class OperationExecutor {
                +run()
                -get_buffer_ptr()
                -compute_library
            }

            class NEONComputeLibrary {
                +create_tensor()
                +create_and_copy_tensor()
                +createNormalization()
                +createDequantization()
                +...()
            }

            package "Operators" #888888 {
                class Dequantization {
                    +initialize()
                    +execute()
                }
                class Softmax {
                    +initialize()
                    +execute()
                }
            }
        }
    }
}

UserDriver <|-- CpuUserDriver
IOperationConstructor <|.. OperationConstructor
IOperationExecutor <|.. OperationExecutor
IComputeLibrary <|.. NEONComputeLibrary
CpuUserDriver *-- OperationConstructor
CpuUserDriver *-- OperationExecutor
CpuUserDriver *-- NEONComputeLibrary
OperationConstructor #-- NEONComputeLibrary
OperationExecutor #-- NEONComputeLibrary
OperationConstructor *-- Operators
OperationExecutor *-- Operators
@enduml
```

## CPU UD APIs
#### get_instance()
    CpuUserDriver is created as singleton object.
    ENN engine creates it and then calls UD APIs.
#### Initialize()
    Called from Engine::init().
    Create objects of NEONComputeLibrary, OperationConstructor and OperationExecutor.
#### OpenSubGraph()
    Called from Engine::open_model().
    Uses OperationConstructor and initializes each operator in OperatorList.

|Type|Name|Description|
|---|---|---|
|std::shared_ptr<<model::component::OperatorList>>|operator_list|The operator list created with Graph composed of vertex and edge|
|UdSubGraphPreference&|preference|Not used in CPU UD|

#### ExecuteSubGraph()
    Called from Engine::execute_model().
    Get UDOperators from a map managed in CPU UD.
    Uses OperationExecutor and set buffers with each buffer index from BufferTable.
    Then executes each operator.

|Type|Name|Description|
|---|---|---|
|uint64_t|operator_list_id|An unique id created in OpenSubGraph to manage operator list|
|model::memory::BufferTable&&|buffer_table|The indexed map composed input/output buffer of each operator|

#### CloseSubGraph()
    Called from Engine::close_model().
    Removes UDOperators from the map.

|Type|Name|Description|
|---|---|---|
|uint64_t|operator_list_id|An unique id created in OpenSubGraph to manage operator list|

#### Deinitialize()
    Called from Engine::deinit().
    Clear the UDOperators' map.

## CPU UD call sequence
```plantuml
@startuml
scale 800 width

actor "ENN Engine"
participant "CPU UD"
participant "OperationConstructor"
participant "NEONComputeLibrary"
participant "OperationExecutor"
participant "Any Operator"

"ENN Engine" -> "CPU UD": Initialize()
"CPU UD" -> "CPU UD": create object of NEONComputeLibrary
"CPU UD" -> "CPU UD": create object of OperationConstructor
"CPU UD" -> "CPU UD": create object of OperationExecutor
"CPU UD" --> "ENN Engine"
... (process) ...
"ENN Engine" -> "CPU UD": OpenSubGraph()
"CPU UD" -> "OperationConstructor": initialize_ud_operators
"OperationConstructor" --> "CPU UD"
"CPU UD" -> "OperationConstructor": create_ud_operator<T>
"OperationConstructor" -> "OperationConstructor": get options or parameters
"OperationConstructor" -> "OperationConstructor": create in/out tensors from featuremap
"OperationConstructor" -> "NEONComputeLibrary": create operator object
"NEONComputeLibrary" -> "Any Operator": constructor
"Any Operator" --> "NEONComputeLibrary"
"NEONComputeLibrary" --> "OperationConstructor"
"OperationConstructor" -> "Any Operator": initialize
"Any Operator" --> "OperationConstructor"
"OperationConstructor" --> "CPU UD"
"CPU UD" -> "OperationConstructor": get_ud_operators
"OperationConstructor" --> "CPU UD"
"CPU UD" -> "CPU UD": add operators in map
"CPU UD" --> "ENN Engine"
... (process) ...
"ENN Engine" -> "CPU UD": ExecuteSubGraph()
"CPU UD" -> "CPU UD": get operators from map
"CPU UD" -> "OperationExecutor": run
"OperationExecutor" -> "OperationExecutor": set in/out buffer
"OperationExecutor" -> "Any Operator": execute
"Any Operator" --> "OperationExecutor"
"OperationExecutor" --> "CPU UD"
"CPU UD" --> "ENN Engine"
... (process) ...
"ENN Engine" -> "CPU UD": CloseSubGraph()
"CPU UD" -> "CPU UD": remove operators from map
"CPU UD" --> "ENN Engine"
... (process) ...
"ENN Engine" -> "CPU UD": Deinitialize()
"CPU UD" -> "CPU UD": clear operators map
"CPU UD" --> "ENN Engine"

@enduml
```