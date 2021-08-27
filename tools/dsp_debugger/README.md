---
Title: dsp_debugger
Output: pdf_document
author: Hoon Choi
Date: 2021. 8. 11
Reference: http://pdf.plantuml.net/PlantUML_Language_Reference_Guide_ko.pdf
---

## dsp_debugger.py
> written by hoon98.choi (2021-08-11)

### 1. Introduction
With DSP Debugger tool, the model developers can verify DSP layer-level.
User can test each layer as well as part of layer lists.


### 2. Requirements
* EnnTest_v2_service or EnnTest_v2_lib
  * EnnLibraries on Target
* python 3.x
* python-yaml
  * $ sudo apt install python-yaml   // ubuntu only
  * $ python -m pip install pyyaml

### 3. Components
#### 3-0. Materials example
```bash
├── dsp_debugger.py
├── dsp_iv3
│   ├── dsp_input_0.bin
│   ├── dsp_iv3_v4_dirty_test_0_31_107_ref.sci
│   ├── dsp_iv3_v4.sci
│   ├── dsp_output_0.bin
│   ├── iv3_dirty.nnc
│   ├── iv3.nnc
│   ├── nnc_input_0.bin
│   └── nnc_output_0.bin
├── print_util.py
└── README.md
```
#### 3-1. .sci
  - scenario file to use debugger
  - The information is described with .yaml syntax
```yaml
Example: dsp_iv3/dsp_iv3_v4.scl

dsp_debug_scenario:
    name: IV3_cmdq_test
    local:
      model_file: dsp_iv3/iv3.nnc  # relative path to push files to device
      input:
        - dsp_iv3/nnc_input_0.bin
        - dsp_iv3/nnc_input_1.bin  # If inputs are multiple
        - dsp_iv3/nnc_input_2.bin
      golden:
        - dsp_iv3/nnc_output_0.bin
    target:
      model_file: /data/raw/iv3.nnc # model file in target, includes absolute path
      input:
        - /data/raw/nnc_input_0.bin
        - /data/raw/nnc_input_1.bin
        - /data/raw/nnc_input_2.bin
      golden:
        - /data/raw/nnc_output_0.bin
    max_layer: 108                  # The Maximum number of layer. 108 is the last
```

#### 3-2. Others
* **Compiled model file** with reference intermediate buffers
* binary files
  * input binaries
  * golden binaries to verify output binary


### 4. Usage Guide
1. -p : push all materials from local to device
2. -s : scenario file
  - User can set input / output / model files each with parameters
3. -i : inputs
  - User can set multiple inputs 
    - ex) -i in1.bin in2.bin in3.bin ...
4. -g : golden outputs to compare with output
  - User can set multiple outputs 
    - ex) -g gold1.bin gold2.bin
5. --dbg_all : execute and verify **once** from start layer to end layer
6. -S : set start layer number
7. -E : set end layer number
8. --debug : show all messages

Examples)
```
  \$ python3 dsp_debugger.py -f my_debug_model.nnc -i input1.dat input2.dat -g gold1.dat gold2.dat -m 100 --dbg_all
  \$ python3 dsp_debugger.py -s scenario/my_scenario.sci --dbg_all
  \$ python3 dsp_debugger.py -s scenario/my_scenario.sci --start 0 --finish 1
  \$ python3 dsp_debugger.py -s dsp_iv3/dsp_iv3_v4_dirty_test_0_31_107_ref.sci --dbg_all -S 2 -F 4
```

Examples of execution)
* windows
```bash
D:\...dsp_debugger> python dsp_debugger.py -s dsp_iv3/dsp_iv3_v4_dirty_test_0_31_107_ref.sci -S 1 -F 4
 # Print Release Mode
 # -----------------------------------------
 # 1. Scenario Name: IV3_cmdq_test
 # 2. local_model_file: dsp_iv3/iv3_dirty.nnc
 # 3. local_input: ['dsp_iv3/nnc_input_0.bin']
 # 4. local_golden: ['dsp_iv3/nnc_output_0.bin']
 # 5. target_model_file: /data/raw/iv3_dirty.nnc
 # 6. target_input: ['/data/raw/nnc_input_0.bin']
 # 7. target_golden: ['/data/raw/nnc_output_0.bin']
 # 8. max number of layer: 108
 # -----------------------------------------
 # Execute one by one from 1 - 4
 # Test layer #1 / #4 .................  FAILED
 # [001-001] Output is DIFFERENT --> save output to dsp_intermediate_dump_1-1.bin
/data/vendor/enn/ref_dsp_intermediate.bin: 1 file pulled, 0 skipped. 26.0 MB/s (12157440 bytes in 0.446s)
/data/vendor/enn/dsp_intermediate.bin: 1 file pulled, 0 skipped. 18.0 MB/s (12157440 bytes in 0.644s)
 # Test layer #2 / #4 .................  SUCCESS
 # Test layer #3 / #4 .................  SUCCESS
 # Test layer #4 / #4 .................  SUCCESS
 #
 # Test finished
 # -----------------------------------------
```

* linux
```bash
[hara:~/.../tools/dsp_debugger]$ python3 dsp_debugger.py -s dsp_iv3/dsp_iv3_v4.sci --debug --dbg_all
sh: 1: color: not found
 # Print Debug Mode
 #  [DBG] Input from User: Namespace(brief_show=False, dbg_all=True, execute_once=False, finish=108, golden=[], input=[], local_golden=[], local_inputs=[], local_model_file='', max_layer='108', model_file='', push_materials=False, scenario='dsp_iv3/dsp_iv3_v4.sci', start=0, test_name='layer_by_layer_test', test_one_by_one=True)
 #  [DBG] Scenario is set, we will ignore -i, -g, -f, -m options
 # -----------------------------------------
 # 1. Scenario Name: IV3_cmdq_test
 # 2. local_model_file: dsp_iv3/iv3.nnc
 # 3. local_input: ['dsp_iv3/nnc_input_0.bin']
 # 4. local_golden: ['dsp_iv3/nnc_output_0.bin']
 # 5. target_model_file: /data/raw/iv3.nnc
 # 6. target_input: ['/data/raw/nnc_input_0.bin']
 # 7. target_golden: ['/data/raw/nnc_output_0.bin']
 # 8. max number of layer: 108
 # -----------------------------------------
 #  [DBG] Change start and finish layer to 1, 0
 # Execute_once is set
 #  [DBG] System args: Namespace(brief_show=False, dbg_all=True, execute_once=False, finish=108, golden=['/data/raw/nnc_output_0.bin'], input=['/data/raw/nnc_input_0.bin'], local_golden=['dsp_iv3/nnc_output_0.bin'], local_input=['dsp_iv3/nnc_input_0.bin'], local_inputs=[], local_model_file='dsp_iv3/iv3.nnc', max_layer=108, model_file='/data/raw/iv3.nnc', push_materials=False, scenario='dsp_iv3/dsp_iv3_v4.sci', start=0, test_name='IV3_cmdq_test', test_one_by_one=True)
 #  [DBG] adb_cmd: adb shell "[ -f /data/raw/nnc_input_0.bin ]"
 #  [DBG] adb_cmd: adb shell "[ -f /data/raw/nnc_output_0.bin ]"
 #  [DBG] adb_cmd: adb shell "[ -f /data/raw/iv3.nnc ]"
 #  [DBG] start & end is ignored. execute all layers at once (0 - 108)
 # Execute once from 0 - 108
 # Test #0 ~ #108 ................. 
 #  [DBG] Result:  SUCCESS
 # 
 # Test finished
 # -----------------------------------------

```






### Bash Logs

```bash
usage: dsp_debugger.py [-h] [--test_name TEST_NAME] [--model_file MODEL_FILE] [--input INPUT [INPUT ...]] [--golden GOLDEN [GOLDEN ...]] [--max_layer MAX_LAYER] [--scenario SCENARIO] [--start START] [--finish FINISH]
                       [--dbg_all] [--debug]


* dbg_all is enabled, the debugger try to test ONCE from [START] layer to [FINISH] layer
* dbg_all is disabled(default), the debugger try to test one by one

Version: 202108011

optional arguments:
  -h, --help            show this help message and exit
  --test_name TEST_NAME
                        set name of test
  --model_file MODEL_FILE, -f MODEL_FILE
                        set model file(.nnc)
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        input data, -i <input1> <input2>..
  --golden GOLDEN [GOLDEN ...], -g GOLDEN [GOLDEN ...]
                        golden output data, -o <golden1> <golden2>..
  --max_layer MAX_LAYER, -m MAX_LAYER
                        max number(id) of layer (default: 100)
  --scenario SCENARIO, -s SCENARIO
                        put scenario file(golden, input, model_file, max_number are ignored)
  --start START, -S START
                        start number(id) of layer(default: -1)
  --finish FINISH, -F FINISH
                        finish number(id) of layer(default: -1)
  --dbg_all             debug all(start, finish are ignored)
  --debug               Print with debug logs
  ```
