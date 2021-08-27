---
Title: user sequence diagram with API proposal v0.1
Output: pdf_document
author: Hoon Choi
Date: 2020. 12. 21
Reference: http://pdf.plantuml.net/PlantUML_Language_Reference_Guide_ko.pdf
---
<i>This document simply describe API use cases</i>

### 1. Model LifeCycle

```plantuml
skinparam DefaultTextAlignment center
skinparam defaultFontSize 13
actor User as user #CFE2F3
participant "ENN Framework" as enn
participant "Targets(NPU, DSP,..)" as target
user->enn: Enninitialize()
enn->target: initialize targets
user->enn: EnnOpenModel()
enn->target: parse and download model to target
alt Buffer(memory) related actions
group conventional approach
user->enn: EnnAllocateInputBuffers()
enn->user: buffer list for input
user->enn: EnnAllocateOutputBuffers()
enn->user: buffer list for output
end
group new: get each buffer's information
user->enn: EnnGetBufferShapeByLabel()
enn->user: return buffer shape with label
user->enn: EnnGetBufferShapeByIndex()
enn->user: return buffer shape with label
end
group new: Create own buffer
user->enn: EnnCreateBuffer()
enn->user: return buffer
end
group new: Import Export Buffer
user->enn: EnnCreateBufferFromFd()
enn-->user: return buffer
user->enn: EnnCreateBufferFromFdWithOffset()
enn-->user: return buffer
end
group new: Set buffer to model
user->enn: EnnSetBufferByIndex()
enn->user: return result
user->enn: EnnSetBufferByLabel()
enn->user: return result
end
end
group new: Get model information
user->enn: EnnGetModelInfo(COMPILER_VER)
user->enn: EnnGetModelInfo(MODEL_VER)
user->enn: EnnGetModelInfo(MODEL_DESC)
enn-->user: (return with string parameter)
end
group new: Model verification
user->enn: EnnVerifyModel()
enn->user: return result
end
user->enn: EnnExecuteModel()
enn->target: Send Execute Command with buffer information
target-->user: return result
user->enn: EnnReleaseBuffers()
enn->user: return result
user->enn: EnnCloseModel()
enn->user: return result
user->enn: EnnDeinitialize()
enn->user: return result


```
<br>
### 2. Initialize / Deinitialize

```plantuml
@startuml
skinparam DefaultTextAlignment center
skinparam defaultFontSize 13
actor User as user #CFE2F3
participant "User API" as api
participant ServiceInterface as if
participant "Service Framework" as service
participant "Targets" as target


user->api: EnnInitialize()
api->if: enn::client::Initialize()\natomic(ref_n+1)
if->service: enn::core::Initialize()
service->service: (Create Monitor thread)
service->target: initialize targets
target-->user: return result (simplified)

... (process) ...

user->api: EnnDeinitialize()
api->if: enn::client::Deinitialize()\nactomic(ref_n+1)
if->service: enn::core::Deinitialize()
service->target: Deinitialize targets
target-->user: return result (simplified)

@enduml
```

<br>
### 3. EnnOpenModel / EnnCloseModel
```plantuml
@startuml
skinparam DefaultTextAlignment center
skinparam defaultFontSize 13
actor User as user #CFE2F3
participant "User API" as api
participant "Service Framework" as service
participant "Parser & constructor" as parser
participant "Runtime" as rt
participant "Targets" as target

user->api: EnnOpenModel(filename)
group ref_cnt(open) == 0?
api->api: allocate dmabuf buffer and load from file \nref_cnt++
api->service: Open(fd, size, *model_id)
service->parser: Parse(fd, size, *model_id)
parser->parser: Optimize and gen ennModel
note left: Parse, Generate EnnModel, intermediate buffers\nGenerate session_info which requires to client
parser->rt: EnnModel
rt->target: Download {ncp or tsgd} for target
parser->service: session_info*
service->api: session_info
end
target-->user: return result with model_id

... (process) ...

user->api: EnnCloseModel()
api->api: Delete resources \nif (--ref_n) == 0
api->service: Request remove resource(model_id), if ref_cnt == 0
service->rt: Request remove resource(model_id)
rt->target: Request unload binary {ncp or tsgd} to target
target-->user: return result

@enduml
```

####Expected Output (session info):
##### session_info
| type | name | description |
| ---|---|---|
| uint64_t | model_id | model identifier |
| vector<buffer> | buffers | buffer information for user [read only] |
| vector<region> | region | memory information for system [read only] |
<br>
####Buffer
* User view: User API can approach with this data structure
* Read only
* example:

##### buffer
| type | name | example | description |
| ---  | --- | ---| --- |
| int | region_idx | {0, 1, 2.. } | Index of region |
| int | direction | {IN, OUT, NONE} | is this buffer in/output of the graph? |
| int | size | 23823 | size of the buffer |
| int | offset | 123 | offset of the buffer : size+offset should be less then region[region_idx] size |
| shape_info | n x w x h x c | {1, 300, 400, 3} | if n >= 1, we can guess this buffer is array |
| int | buffer_type | {U8, RGBD, F32.. } | enum which defined in header |
| string | name | input_0 |name of edge(buffer). User can access memory with the name |
<br>
####Region
  * view as a system: the framework manage this and execute model with parameter generated from this data structure

##### region

| type | name | example | description |
|---|---|---|---|
| int  | attr | {MENDATORY \| IS_FD..} | contain multiple attributes: MANDATORY, IS_FD, if MANDATORY is not set, the user can skip the index, IS_FD means that this is used by HIDL |
| int | required_size | 10000 | size of the memory. {size + offset} should not be bigger than actual size |
| string | name | TEMP | name of region for debug or internal usage |
<br>
####Region as a parameter of execution_model()
  * Region as a parameter (user >> framework)
  * example
```c++
  /* 1. single execution */
  enn_execute_model(model_id, exec_region);  // execute once with exec_region

  /* 2. multiple execution */
  enn_memory_commit(model_id, vec<exec_region>);  // prepare region set to service framework
  enn_execute_model_with_region_idx(model_id, region_id); // execute with commited exec_region
```
##### exec_region

| type | name | example | description |
|---|---|---|---|
| int | attr | {BLANK, IS_FD..} | Attribute. If BLANK is set, this index is ignored by the framework. this can be set if MENDATORY is not set in returned session_info | 
| int  | fd | {0, 1, 2.. } | file descriptor |
| addr_t | address | 0x82372311dd00 | if libmode, address will be sent without IS_FD type |
| int | size | 10000 | size of the memory. {size + offset} should not be bigger than actual size |
| int | offset | 123 | start point of value |


<br>
### 4. Memory related actions
*> if you want to recent usage, you can refer api_test_sample_?_blabla in enn_api-test.cc*

> Basically, User set executable buffers with 3 stages
> 1. Set Ext-buffers: user can manage simple buffer array
> > ```c++
> >    EnnBuffer** ext_buf_array;
> >    // Array must be managed by in-out-ext order
> >    // This should be same with you returns after calling EnnAllocateAllBuffers()
> > ```
> 2. Update Ext-buffers to Model Container
> > ```c++
> >    EnnSetBuffers(model_id, ext_buf_array, (n_in + n_out + n_ext));
> >    EnnSetBufferByIndex(model_id, (IN or OUT), index, buffer);
> >    EnnSetBufferByLabel(model_id, label, buffer);
> > ```
> 3. Commit buffers to register to service runtime (model manager)
> > ```c++
> >    EnnBufferCommit(model_id);
> > ```
```plantuml
@startuml
skinparam DefaultTextAlignment center
skinparam defaultFontSize 13
actor User as user #lightgray
participant "User API" as api
participant "Memory Manager" as mm
participant "Service Framework" as service
participant "Targets" as target

== case 1: Conventional Usage ==
user->api: EnnAllocateInputBuffers()
api<->mm: Request / response allocate
api->mm: request allocation with libion
mm-->api: return result
api-->user: return buffer array
user->api: EnnAllocateOutputBuffers()
api->mm: request allocation with libion
mm-->api: return result
api-->user: return buffer array

== case 2: Internal buffer ==
user->api: EnnGetBufferShapeInModelByIndex(m_id, IN, 0)
api-->user: return shape {n, w, h, c, bpp}
user->user: calculate buffer size
user->api: EnnCreateBuffer(size)
api->mm: request allocation with libion
mm-->api: return result
api-->user: return result (Buffer allocated)

== case 3: External Buffer ==
user->api: EnnGetBufferShapeInModelByLabel(m_id, label)
api-->user: return shape {n, w, h, c, bpp}
->user: Get external\nmemory
user->user: Validate memory size
user->api: EnnCreateBufferFromFd(fd, size)
api->mm: Register fd and size
mm->api: return virtual address
api-->user: return result (EnnBuffer)

== case 4: External Partial Buffer ==
->user: Get external\nmemory
user->api: EnnCreateBufferFromFdWithOffset(fd, size, offset)
api->mm: Register fd, size and offset
mm->api: return virtual address
api-->user: return result (EnnBuffer)
user->api: EnnCreateBufferFromFdWithOffset(fd, size, offset)
api->mm: Register fd, size and offset
mm->api: return virtual address
api-->user: return result (EnnBuffer)

@enduml
```

<br>
### 5. Execution Model // to be updated

```plantuml
@startuml
skinparam DefaultTextAlignment center
skinparam defaultFontSize 13
actor User as user #pink
participant "User API" as api
participant "Memory Manager" as mm
participant "Service Framework" as service
participant "Master Controller" as ma
collections "execution threads" as exethread
collections "Target drivers" as td
collections "Targets" as target

== Case 1: Set buffers and execution ==
user->api:EnnSetBufferByIndex(m_id, dir, idx, buf)
user->api:EnnSetBufferByLabel(m_id, Label, buf)
group option 1: send buffer to service with setBuffer API
api->service:enn::core::requestRegisterBuffer()
note left: Requires a little time\nto transfer buffer through RPC
service->service:If already set another buf, \noverwrite it
service->service:set ref_cnt of buffer
service-->api: return result
end
api->mm: request buffer to set ref_cnt
mm-->api: return result
user->api: EnnVerifyModel()
api->api: Check if buffers\nare registered
user->api: EnnExecuteModel()\n(For sync mode)
api->api: If not verified,\ncall EnnVerifyModel()
group option 2: send buffers with execution cmd
api->api: prepare <i>session</i> user registered before
end
api->service: Call enn::core::execution_model(session)
== case 2: Execution with Buffers ==
user->api: EnnExecuteModelWithBuffers*(buffer_list)
api->api: Call EnnVerifyModelWithParameters(buffer_list)
api->api: prepare <i>session</i> with <i>buffer_list</i>
api->mm: request buffer to set ref_cnt
mm-->api: return result
api->service: Call enn::core::execution_model(session)
note left: Requires a little time\nto transfer buffers through RPC
== Common: Execution process ==
service->ma: Execution Model(EnnModel, session)
loop execute model
ma-->exethread: Distribute sub-task \nto execution threads
exethread-->td: call exec function \nto target drivers
td->target: send exec \ncmd to target
target-->td: return result
td->exethread: return result
exethread->exethread: wait or trigger to \nanother executer thread
exethread-->ma: return done from the \nlast sub-task executers
end
ma->ma: Check the model is done
ma->service: return result
service->api: return result
api->user: return result
@enduml

```

* *<i>EnnExecuteModelWithBuffers(buffer_list)</i> and <i>EnnVerifyModelWithParamters()</i> will be used if asynchronous mode is supported

<br>
### 6. Other Usages

```plantuml
@startuml
skinparam DefaultTextAlignment center
skinparam defaultFontSize 13
actor User as user #pink
participant "User API" as api
participant "Service Framework" as service

user->api: EnnLoadModel()
api<-->service: (load)
api-->user: return result
user->api: EnnGetModelInfo(RELEASED_DATE)
api-->user: ex. "2020.12.31"
user->api: EnnGetModelInfo(COMPILER_VER)
api-->user: ex. "1.5.4.32"

@enduml
```
