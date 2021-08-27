# -*- coding: utf-8 -*-
#!/usr/bin/env python

import argparse
import sys
import os
import textwrap
import print_util
import yaml
import filecmp
import shutil
import platform

def check_validity(args):
    # in != out
    if len(args.input) == 0:
        print_util._print_red("Error: length of input(%d), golden(%d)"%(len(args.input), len(args.golden)))
        return False
    if check_file_exist_in_target(args.input) == False:
        return False
    if check_file_exist_in_target(args.golden) == False:
        return False
    if check_file_exist_in_target([args.model_file]) == False:
        return False
    return True


def check_file_exist_in_target(file_vectors):
    for f in file_vectors:
        adb_cmd = "adb shell \"[ -f %s ]\""%f
        print_util._print("adb_cmd: %s"%adb_cmd)
        ret = os.system(adb_cmd)
        if ret != 0:
            print_util._print_red("Error: File not found: %s"%f)
            return False
    return True

def check_file_exist_in_local(file_vectors):
    for f in file_vectors:
        if os.path.isfile(f) == False:
            print_util._print_red("Error: File not found: %s"%f)
            return False
    return True


def load_file_to_get(filename):
    if check_file_exist_in_local([filename]) == False:
        return False

    with open(filename) as f:
        try:
            file_load_buffer = yaml.safe_load(f)
            dsp_debug_scenario = file_load_buffer['dsp_debug_scenario']
            _name = dsp_debug_scenario['name']
            _local = dsp_debug_scenario['local']
            _target = dsp_debug_scenario['target']
            _local_model_file = _local['model_file']
            _local_input = _local['input']
            _local_golden = _local['golden']
            _target_model_file = _target['model_file']
            _target_input = _target['input']
            _target_golden = _target['golden']

            _max_layer = dsp_debug_scenario['max_layer']
        except Exception as e:
            print_util._print_red("File Open Failed: " + filename)

    print_util._print_green("-----------------------------------------")
    print_util._print_green("1. Scenario Name: %s"%_name)
    print_util._print_green("2. local_model_file: %s"%_local_model_file)
    print_util._print_green("3. local_input: %s"%_local_input)
    print_util._print_green("4. local_golden: %s"%_local_golden)
    print_util._print_green("5. target_model_file: %s"%_target_model_file)
    print_util._print_green("6. target_input: %s"%_target_input)
    print_util._print_green("7. target_golden: %s"%_target_golden)
    print_util._print_green("8. max number of layer: %d"%_max_layer)
    print_util._print_green("-----------------------------------------")

    return _name, _local_model_file, _local_input, _local_golden, _target_model_file, _target_input, _target_golden, _max_layer


def set_cmd_selinux(val):
    return "setenforce %d"%val


def set_cmd_test_v2(args, check_ref):
    input_str = ' '.join(args.input)
    golden_str = ' '.join(args.golden)
    adb_cmd = "/vendor/bin/EnnTest_v2_lib --model %s --input %s --golden %s &> /dev/null"%(args.model_file, input_str, golden_str)
    return adb_cmd

'''
    if check_ref:
        print_util._print_green("check reference")
    else:
        print_util._print_green("no check")
'''

def set_boundary(args):
    if args.dbg_all == True:
        args.test_one_by_one = False
        print_util._print("start & end is ignored. execute all layers at once (%d - %d)"%(args.start, args.finish))
    elif args.start == -1 and args.finish == -1:
        args.start = 0    # Start Number of Layer
        args.finish = args.max_layer   # Max Number of Layer
        print_util._print("start & end is not set: means, full execution layer by layer (%d - %d)"%(args.start, args.finish))


def set_cmd_boundary(start, finish):
    return "echo %d %d > /sys/kernel/debug/npu/layer_range"%(start, finish)


def set_debug_mode(sw):
    if sw == True:
        return "setprop vendor.enn.dsp.lbl 1"
    else:
        return "setprop vendor.enn.dsp.lbl 0"


def make_exec_command(args, start, finish, check_ref):
    cmd_set = []
    cmd_set.append(set_cmd_selinux(0))
    cmd_set.append(set_debug_mode(True))
    cmd_set.append(set_cmd_boundary(start, finish))
    cmd_set.append(set_cmd_test_v2(args, check_ref))
    cmd_set.append(set_cmd_boundary(0, args.max_layer))
    cmd_set.append(set_debug_mode(False))

    return cmd_set


def execute_adb_shell(var, brief_show):
    adb_cmd = "adb shell \"%s"%var + "\""
    if brief_show == True:
        if platform.system() == "Windows":
            adb_cmd += " > NUL"
        else:
            adb_cmd += " > /dev/null"

    if os.system(adb_cmd):
        print_util._print_red("adb command execution failed: %s"%adb_cmd)
        return False
    return True


def adb_push_files(args):
    if args.local_model_file == "" or len(args.input) == 0 or len(args.golden) == 0:
        print_util._print_red("Please use scenario file mode if you want to push from local PC")
        return False
    adb_cmd = "adb push %s %s"%(args.local_model_file, args.model_file)
    print_util._print("adb_cmd: %s"%adb_cmd)
    if os.system(adb_cmd):
        print_util._print_red("adb command execution failed: %s"%adb_cmd)
        return False

    for idx in range(len(args.local_input)):
        adb_cmd = "adb push %s %s"%(args.local_input[idx], args.input[idx])
        print_util._print("adb_cmd: %s"%adb_cmd)
        if os.system(adb_cmd):
            print_util._print_red("adb command execution failed: %s"%adb_cmd)
            return False

    for idx in range(len(args.local_golden)):
        adb_cmd = "adb push %s %s"%(args.local_golden[idx], args.golden[idx])
        print_util._print("adb_cmd: %s"%adb_cmd)
        if os.system(adb_cmd):
            print_util._print_red("adb command execution failed: %s"%adb_cmd)
            return False

    return True


def execute_adb_root():
    os.system("adb root")


def execute_adb_pull(filename):
    adb_cmd = "adb pull \"%s\""%filename
    print_util._print("adb cmd: %s"%adb_cmd)
    if os.system(adb_cmd):
        print_util._print_red("adb file pull failed: %s"%filename)
        return False
    return True

def execute_adb_diff(filename1, filename2):
    adb_cmd = "adb shell \"diff %s %s > /dev/null\""%(filename1, filename2)
    if os.system(adb_cmd):
        return False
    return True


def check_file_difference(ref, cmp, _start, _finish):
    print_util._print("Result: ", '')
    if execute_adb_diff(ref, cmp) == False:
        print_util._print_red("FAILED", '\n', '')
        diff_filename = "dsp_intermediate_dump_%d-%d.bin"%(_start, _finish)
        print_util._print_red("[%03d-%03d] Output is DIFFERENT --> save output to %s"%(_start, _finish, diff_filename))
        execute_adb_pull(ref)
        execute_adb_pull(cmp)
        shutil.move("dsp_intermediate.bin", diff_filename)
        return False
    print_util._print_green("SUCCESS", '\n', '')
    return True


def execute_and_verify_test(args, _start, _finish, check_ref):
    ref_file = "/data/vendor/enn/ref_dsp_intermediate.bin"
    cmp_file = "/data/vendor/enn/dsp_intermediate.bin"
    cmd_set = make_exec_command(args, _start, _finish, check_ref)

    cmd_var = ""
    for cmd in cmd_set:
        cmd_var += cmd + ";"
    if execute_adb_shell(cmd_var, args.brief_show) == False:
        print_util._print_red("FAILED")
        return False
    diff_ret = check_file_difference(ref_file, cmp_file, _start, _finish)


def run(args):
    if args.brief_show:
        os.environ['enn_test_brief_show'] = str(True)
        print_util._print_green("Print Release Mode")
    else:
        print_util._print_green("Print Debug Mode")
    print_util._print("Input from User: %s"%args)

    if args.scenario != '':
        print_util._print("Scenario is set, we will ignore -i, -g, -f, -m options")
        try:
            args.test_name, args.local_model_file, args.local_input, args.local_golden, args.model_file, args.input, args.golden, args.max_layer = load_file_to_get(args.scenario)
        except:
            print_util._print_red("Scenario may be wrong")
            return False

    if args.dbg_all == True:
        print_util._print("Change start and finish layer to 1, 0")
        print_util._print_green("Execute_once is set")

    print_util._print("System args: %s"%args)

    if args.push_materials == True:
        if adb_push_files(args) == False:
            return False

    if check_validity(args) == False:
        print_util._print_red("Check Validity Failed!")
        return False

    set_boundary(args)

    # Execute
    # execute_adb_root()

    # if dbg_all?
    if args.test_one_by_one == False:
        print_util._print_green("Execute once from %d - %d"%(args.start, args.finish))
        print_util._print_green("Test #%d ~ #%d ................. "%(args.start, args.finish), "")
        print_util._print("", "\n", "")
        execute_and_verify_test(args, args.start, args.finish, True)
    else:
        print_util._print_green("Execute one by one from %d - %d"%(args.start, args.finish))
        for idx in range(args.start, args.finish + 1):
            print_util._print_green("Test layer #%d / #%d ................. "%(idx, args.finish), "")
            print_util._print("",'\n',"")
            execute_and_verify_test(args, idx, idx, False)

    print_util._print_green("")
    print_util._print_green("Test finished")
    print_util._print_green("-----------------------------------------")

    return True


def main(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
    # Version: 20210805

    # Example Usage:
       $ python3 dsp_debugger.py -f my_debug_model.nnc -i input1.dat input2.dat -g gold1.dat gold2.dat -m 100 --dbg_all
       $ python3 dsp_debugger.py -s scenario/my_scenario.sci --dbg_all
       $ python3 dsp_debugger.py -s scenario/my_scenario.sci --start 0 --finish 1

    '''))
    parser.add_argument('--test_name', type=str, default='layer_by_layer_test',
                        help='set name of test')
    parser.add_argument('--model_file', '-f', type=str, default='',
                        help='set model file(.nnc)')
    parser.add_argument('--input', '-i', nargs='+', default=[],
                        help='input data, -i <input1> <input2>..')
    parser.add_argument('--golden', '-g', nargs='+', default=[],
                        help='golden output data, -o <golden1> <golden2>..')
    parser.add_argument('--max_layer', '-m', type=str, default='108',
                        help='max number(id) of layer (default: %(default)s)')
    parser.add_argument('--scenario', '-s', type=str, default='',
                        help='put scenario file(golden, input, model_file, max_number are ignored)')
    parser.add_argument('--start', '-S', type=int, default='0',
                        help='start number(id) of layer(default: %(default)s)')
    parser.add_argument('--finish', '-F', type=int, default='108',
                        help='finish number(id) of layer(default: %(default)s)')
    parser.set_defaults(execute_once=False)
    parser.add_argument('--dbg_all', dest='dbg_all', action='store_true',
                        help='debug all(start, finish are ignored)')
    parser.set_defaults(brief_show=True)
    parser.add_argument('--debug', dest='brief_show', action='store_false',
                        help='Print with debug logs')
    parser.add_argument('--push', '-p', dest='push_materials', action='store_true',
                        help='push all materials')
    parser.set_defaults(dbg_all=False)
    parser.set_defaults(test_one_by_one=True)
    parser.set_defaults(push_materials=False)
    parser.set_defaults(local_model_file="")
    parser.set_defaults(local_inputs=[])
    parser.set_defaults(local_golden=[])
    args = parser.parse_args()
    if (run(args) == False):
        print_util._print_red("Failed to run")

if __name__ == "__main__":
    main(sys.argv)