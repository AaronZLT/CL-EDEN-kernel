---
Title: Memory Commit and management
Output: pdf_document
author: Hoon Choi
Date: 2021. 4. 28
Reference: http://pdf.plantuml.net/PlantUML_Language_Reference_Guide_ko.pdf
---


## 1. What is memory commit?
> To reduce execution overhead, we added a new step between loadModel and ExecuteModel : COMMIT. In the commit step, memory buffers allocated in user / client side are transferred to Service core and devices. Because transfering and map for memory takes quite long, this step can make execution faster.

## 2. Memory related APIs
> LoadModel() returns memory buffer information and requirement from service. For example, if inception_v3 model is loaded successfully the service(parser) returns the following:
```plantUML
skinparam DefaultTextAlignment center
skinparam defaultFontSize 13
participant "Client/User API" as client
participant "Service/Parser" as service

client->service: LoadModel(model_file.nnc)
service->client: model ID, set(buffer), set(region)

```

```bash
# Session.Model_ID: 0x5D5B000300000000 / 0x5D5B000300000000(buf: 4, reg: 4)::::
# [Buffer] Region_idx(0), dir(0), buf_index(0), size(268203), offset(0),
           shape nwhc(1 x 299 x 299 x 3), buffer_type: 3, name: IFM
# [Buffer] Region_idx(1), dir(2), buf_index(0), size(1000), offset(0),
           shape nwhc(1 x 1 x 1 x 1000), buffer_type: 9, name: 691
# [Buffer] Region_idx(2), dir(2), buf_index(1), size(4000), offset(0),
           shape nwhc(1 x 1 x 1 x 1000), buffer_type: 0, name: 689
# [Buffer] Region_idx(3), dir(1), buf_index(0), size(4000), offset(0),
           shape nwhc(1 x 1 x 1 x 1000), buffer_type: 0, name: 690

# [Region] attr(0), req_size(268203), name()
# [Region] attr(0), req_size(1000), name()
# [Region] attr(0), req_size(4000), name()
# [Region] attr(0), req_size(4000), name()
```
Return values are categories into two section:
 - Buffer: Client provides buffer information to user with buffer information
 - Region: Client must provide to service memory buffer with req_size


## 3. Memory commit related APIs
> After load, user can query to client that which buffer should I make?

### 3-1. Queries
 - A. EnnGetBuffersInfo(model_id, &buffer_info)
    - Get # of required buffers (input, output, extra)
 - B. EnnGetBufferInfoByIndex(model ID, dir[i / o / e], index, *output)
    - Get Buffer information from loaded return.
 - C. EnnGetBufferInfoByLabel(model ID, char* label, *output)
    - The user also get buffer info by label.

> output data structure of query APIs
```c++
typedef struct _ennBufferInfo {
    bool     is_able_to_update;
    uint32_t n;  // batch size
    uint32_t width;
    uint32_t height;
    uint32_t channel;
    uint32_t size;
    const char *label;
} EnnBufferInfo;
```

### 3-2. Allocate methods
 ##### A. EnnCreateBufferCache(size), EnnCreateBuffer(size)
   - Allocate cachable or non-cachable buffer (ION/dmabuf case)
   - Retuns EnnBuffer*
 ##### B. EnnCreateBufferFromFd(fd, size), EnnCreateBufferFromFdWithOffset(fd, size, offset)
   - Import fd from ION or dmabuf and map to virtual address
   **- IMPORTANT: after calling this API, va should be calculated by adding offset**
   - ex) address: User can access at the first byte to access [out->va + out->offset]
 ##### C. EnnCreateBufferObject(fd, size, offset)
   - Just wrap buffer with fd, size, offset. Internally doesn't call allocators
 ##### D. EnnReleaseBuffer(EnnBuffer*)

> EnnBuffer : This buffer cannot be modified. Internally base address of the buffer is used to convert to std::shared_ptr<EnnBufferCore> object.
```c++
typedef struct _ennBuffer {
    void *va;
    uint32_t size;  // requested size
    uint32_t offset;
} EnnBuffer;
```

### 3-3. Commit / Session management
> Client context manager manages "session space" which contains buffer set for execution at every model_id.
> Session space is more than 1 if user calls 
 ##### A. EnnGenerateBufferSpace(model ID, # of sets)
   - If this function is not called before calling commit or something, the system generates 1 session space internally.
   - ToDo: check modelContainer->GenerateInferenceData() to get multiple sets
 ##### B. EnnSetBufferByIndex(model ID, direction, index, buf)
   - Set MemoryObject at {direction, index} in session #0
 ##### C. EnnSetBufferByLabel(model ID, label, buf)
   - Set MemoryObject at {label, index} in session #0
 ##### D. EnnSetBuffers(model ID, *bufs, # of total buffers)
   - set all buffers to session #0
   - buffers should be set in { in #0, in #1,.. out #0, out #1,.. ext #0, ext #1,.. ext #n } order
 ##### E. EnnBufferCommit(model ID)
   - transfer commit buffers in session #0
   - if buffer is not completely set, returns error
   - if buffer is sucessfully commited to service, returns success
   - Internally, session sets execution model ID from service


### 4. Scenario
```plantUML
skinparam DefaultTextAlignment center
skinparam defaultFontSize 13
participant "User API" as user
participant "Client(context)" as client
participant "Service/Parser" as service

user->client: initialize()
group LoadModel
user->client: LoadModel(.nnc)
client->service: LoadModel(.nnc)
service->client: model ID, set(buffer), set(region)
client->client: set session data with map::[model ID] = {buffer, region}
client-->user: model ID
end
group getBufferInfo
user-->service: EnnGetBuffersInfo(model ID)
user-->service: EnnGetBufferInfoByIndex(model ID, IN, 0)
service-->user: EnnBufferInfo output
end
group CreateMemoryObjects
user->client: EnnCreateMemory()
user->client: EnnCreateMemoryFromFd()
user->client: EnnCreateMemoryFromFdWithOffset()
user->client: EnnCreateMemoryCache()
user->client: EnnCreateMemoryObject()
client->client: Memory Manager generates std::shared_ptr<EnnBufferCore>
client->user: convert to EnnBuffer* for user view
end
group Generate Session Space and Set Memory Objects
user->client: EnnGenerateBufferSpace(model ID, default 1)
client->client: allocate buffer spaces
user->client: EnnSetBufferByIndex(model ID, direction, index, buf)
user->client: EnnSetBufferByLabel(model ID, label, buf)
user-->client: (Set buffers)
client->client: register buffer at session #0
client-->user: return result
user->client: EnnBufferCommit(model ID)
client->client: verify Session Data at map::[model ID] is corretly set or not
client->service: send commit data to service
service->service: register buffers
service-->client: return execute_model_id
client->client: register execute_model_id to session data
client-->user: return result
end
group Execution
user->client: execution(model ID, session ID = 0)
client->client: find execute_model_id in session space (commit is completed?)
client->service: execute_model([execute_model_id]) // array
service->service: execute with execute_model_id
service-->client: result
client-->user: result
end
```



## 4. TO DO
 #### 4-1. Support multiple session to commit
  - Allow multiple sessions
  - User can select session ID
   ex) EnnGenerateBufferSpace(model ID, # of sessions)
   ex) EnnSetBufferByIndex(model ID, direction, index, buf, **session idx**)
   ex) EnnSetBufferByLabel(model ID, label, buf, **session idx**)
   ex) EnnSetBuffers(model ID, *bufs, # of total buffers, **session idx**)
   ex) EnnBufferCommit(model ID, **session idx**)
 
 #### 4-2. Release commit
  - User can release commit
   ex) EnnReleaseCommit(model_id, **session idx**)
    - release commit and clear execute_model_id at session space

 #### 4-3. Execution Model with multiple session
  - Send all registered execute_model_id in array
   ex) Execute Model (model_id) --> send all registered execute_model_id in array

 #### 4-4. fault tolerance
  - If user delete buffer before commit session, the session should return error
  - Should find uncovered scenarios.

