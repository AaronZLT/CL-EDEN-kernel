---
Title: .nnc displayer (.tflite for Eden)
Output: pdf_document
author: Hoon Choi
Date: 2021. 1. 18
Reference: http://pdf.plantuml.net/PlantUML_Language_Reference_Guide_ko.pdf
---

> This tool tries to show .nnc file for Eden Framework.
You should update fbs_generated.h or similar if schema of .fbs is updated in eden_core.

### What is this?
* This tool tries to analize and show .nnc file (a.k.a .tflite for Eden) in console
* just build with your compiler, and try to execute with .nnc file

### Where to Get
```bash
$ cd source/tools; ./tl_displayer_x86  # executable for linux(ubuntu)
$ cd source/tools/eden_nnc_displayer; ./build.sh; ./tl_displayer_x86
```
> You can change g++ compiler in build.sh for your own environment, even in your device
when compiling with arm-linux-android-g++

### How to Use
```bash
$ ./tl_displayer_x86 <.nnc file>
```


### Example to show
@import "../../doc/materials/20210119_tl_displayer.jpeg"

#### 1. Global description of Model
 - global information of Model
 - Version and Model Description are included at this section

#### 2. Global operator codes
 - Operator codes list used in the model.
 - Opcode has its own enumeration, for example, 25 is SOFTMAS
 - For customization, "32" is used for extension like NCP, etc.

#### 3. Subgraph 0: In/Out tensors, subgraph name
 - .nnc basically can have 2 or more subgraph. 
   - Currently there's only one subgraph for all solutions
 - Input and Output tensors are described in the first line
 - .nnc can include each name of subgraph

#### 4. Subgraph 0: Operators
 - This section has all operatos and their input/output tensors
   * NOTE) Tensor indexes are descibed in yellow ( <font color=yellow>yellow</font> )

#### 5. Subgraph 0: Tensors
 - Tensor indicates buffer in section 6(4. Buffers)
   * NOTE) If tensor information includes {size, offset}, it can be more efficient to use
 - Index of tensor is same as yellow numbers in Section 4.
 - Buf_idx means index of buffer which is written in magenta (<font color=magenta> magenta </font>)
 - The size of tensor can be calculated with {type, shape}
   - For example, size of Tensor 1 is sizeof(FLOAT32) * 3 = 12 bytes
   - buffer 1 is 12 bytes. Correct.

#### 6. Global buffers
 - Exception buffer 0, all are descibed in the .nnc file.
 - NOTE)
   - Owner of buffer?: Who will allocate this buffer?
   - Allocation frequency?: When is this buffer allocated? every execution? load once?
   - Reference if buffer is big: For NCP binary case, how about using reference to location of .nnc?
   - Buffer that doesn't care a content as a default. ( current buffer should have "something" )
