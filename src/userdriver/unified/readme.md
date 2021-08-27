# Userdriver(UD) Interface

## NPU UD Interface

- NPU UD inputs a NPU operator list from ENN engine, and sequentially sends a NPU operator of the list to the unified device driver by using VS4L (vision for Linux).
- NPU UD consists of NPU operator list manager and NPU operator marshaller.
- NPU operator list manager has a table the list of operator lists and adds or deletes a operator list in the table.
- NPU operator marshaller marshals a NPU operator by using VS4L.
- NPU UD has six functions, such as Initialize(), OpenSubGraph(), PrepareSubGraph(), ExecuteSubGraph(), CloseSubGraph(), Deinitialize().

```plantuml
@startuml

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

        package "npu" #DDDDDD {
            class NpuUDOperator {
                +init()
                +set()
                +get()
                +deinit()
            }
            class NpuUserDriver {
                +get_instance()
                +Initialize()
                +OpenSubGraph()
                +ExecuteSubGraph()
                +CloseSubGraph()
                +Deinitialize()
            }
        }
    }
}

UserDriver <|-- NpuUserDriver
NpuUserDriver *-- NpuUDOperator
@enduml
```

## NPU UD APIs
ENN engine calls the following APIs.
#### get_instance()
NpuUserDriver instance is created as a singleton object.
#### Initialize()
Engine::init() calls get_instance() which initializes NPU operator marshaller by invoking init_npu().

#### OpenSubGraph()
Engine::open_model() invokes OpenSubGraph() which inputs a list of operators and add the list into the table. It creates a NpuUDOperator including the model_info_t instance. It sequentially calls open_model() API for each NpuUDOperator.
|Type|Name|Description|
|---|---|---|
|std::shared_ptr<<model::component::OperatorList>>|operator_list|The operator list created with Graph composed of vertex and edge|
|UdSubGraphPreference&|preference|A set of preferences of NPU DD|

#### ExecuteSubGraph()
Engine::execute_model() invokes ExecuteSubGraph(). It gets the list of NpuUDOperators, and sequentially calls execute_req() for each NpuUDOperator.

|Type|Name|Description|
|---|---|---|
|uint64_t|operator_list_id|An unique id created in OpenSubGraph() to manage operator list|
|const model::memory::BufferTable&&|buffer_table|The indexed map composed input/output buffer of each operator|

#### CloseSubGraph()
Called from Engine::close_model().
Removes corresponding instance of the list of NpuUDOperators from the table.

|Type|Name|Description|
|---|---|---|
|uint64_t|operator_list_id|An unique id created in OpenSubGraph() to manage operator list|

#### Deinitialize()
Called from Engine::deinit(). Release the table.

## NPU UD call sequence
```plantuml
@startuml
scale 800 width


actor "ENN Engine" as rt
participant "NPU UD" as npuud
participant "NPU Operator Marshaller" as npuopmar
participant "Link" as link
participant "Unified DD" as dd

rt->npuud: Initialize()
npuud->npuud: create a table of NpuUDOpeator lists.
npuud->npuopmar: init_npu()
npuopmar->link: link_init()
... (process) ...

rt->npuud: OpenSubGraph()
npuud->npuud: generate an unique ID and create corresponding list of NpuUDOperators
alt Per-NpuUDOperator operation
npuud->npuud: create and initialize NpuUDOperator
npuud->npuud: add the NpuUDOperator into the list
npuud->npuopmar: open_model()
npuopmar->npuopmar: check if the arguments is valid
alt Loading a NCP file
npuopmar->npuopmar: check if a NCP request is already requested
npuopmar->npuopmar: create a NCP request
npuopmar->npuopmar: allocate ION buffer
npuopmar->npuopmar: load the NCP into the allocated ION buffer
end
npuopmar->link: link_open_model()
link->dd: open()
link->link: initialize vs4l_graph
link->dd: ioctl(VS4L_VERTEXIOC_S_GRAPH) for NCP
link->link: initialize vs4l_format_list for IFM
link->link: initialize vs4l_format_list for OFM
link->dd: ioctl(VS4L_VERTEXIOC_S_FORMAT) for IFM
link->dd: ioctl(VS4L_VERTEXIOC_S_FORMAT) for OFM
link->dd: ioctl(VS4L_VERTEXIOC_S_PARAM) for applying preference
link->dd: ioctl(VS4L_VERTEXIOC_SCHED_PARAM) for passing parameters
link->dd: ioctl(VS4L_VERTEXIOC_STREAM_ON)
end
npuud->npuud: add the list into the table
... (process) ...

rt->npuud: ExecuteSubGraph()
npuud->npuud: get a list from the table by using operator_list_id.
alt Per-NpuUDOperator operation
npuud->npuud: get a NpuUDOperator instance from the list.
npuud->npuud: create and initialize an req_info_t
npuud->npuopmar: execute_req()
npuopmar->npuopmar: check validity of arguments
npuopmar->npuopmar: enqueue UdReq into g_list_npu_req
npuopmar->npuopmar: select and dequeue UdReq
npuopmar->link: link_execute_req()
link->link: get vs4l_container_list for IFM
link->link: get vs4l_container_list for OFM
link->dd: ioctl(VS4L_VERTEXIOC_S_PARAM) for applying preference
link->link: update vs4l_container_list for IFM
link->link: update vs4l_container_list for OFM
link->dd: ioctl(VS4L_VERTEXIOC_QBUF) for IFM
link->dd: ioctl(VS4L_VERTEXIOC_QBUF) for OFM
link->dd: poll()
link->dd: ioctl(VS4L_VERTEXIOC_DQBUF) for IFM
link->dd: ioctl(VS4L_VERTEXIOC_DQBUF) for OFM
end

... (process) ...
rt->npuud: CloseSubGraph()
npuud->npuud: get a list from the table by using operator_list_id.
alt Per-NpuUDOperator operation
npuud->npuud: get NpuUDOperator from the list.
npuud->npuopmar: close_model()
npuopmar->link: link_close_model()
link->dd: ioctl(VS4L_VERTEXIOC_S_PARAM) for applying preference
link->dd: ioctl(VS4L_VERTEXIOC_STREAM_OFF)
link->dd: close()
end

... (process) ...
rt->npuud: Deinitialize()
npuud->npuopmar: npu_shutdown()
npuopmar->link: link_shutdown()
@enduml
```
