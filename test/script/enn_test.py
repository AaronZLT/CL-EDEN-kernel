# -*- coding: utf-8 -*-
#!/usr/bin/env python

import argparse
import csv
# import commentjson
import json
import os
import sys
import statistics

import test_module.test_func as test_func
import test_module.sqe_test as sqe_test
import test_module.e2e_test as e2e_test
import test_module.print_util as print_util

# PEP 257 -- Docstring Conventions
"""This python script to run SQE tests on the unified enn framework.
"""

def check_required_files():
    ret = True
    if not os.path.exists('./test_config'):
        print('Cannot find test_config')
        ret = False

    if not os.path.exists('./test_module'):
        print('Cannot find test_module')
        ret = False

    if not ret:
        sys.exit('\033[31m''Cannot find required files. Run enn_test.py from source/test/script/''\033[0m')

def init_test(device_id, brief_show, board, config_file):
    if brief_show:
        os.environ['enn_test_brief_show'] = str(True)

    print_util._print("Brifely Show: " + os.environ.get('enn_test_brief_show', "Show All"), True)

    check_required_files()

    adb_manager = test_func.ADBManager()
    if device_id:
        adb_manager.set_device_id(device_id)

    e2e = e2e_test.E2E_Test(adb_manager)
    e2e.set_config_file(board, config_file) # e2e is enabled in CI, so should be worked in various boards
    e2e.update_tc_list()

    sqe = sqe_test.SQE_Test(adb_manager)
    e2e.set_config_file(board, None)        # sqe_test only uses sqe_test.yaml
    sqe.update_tc_list()

    return {'e2e': e2e, 'sqe': sqe}


def make_list_from_str(string: str):
    tc_list = []
    string = string.replace('[', '').replace(']', '').replace(' ', '')
    str_list = string.split(',')
    for str_num in str_list:
        tc_list.append(int(str_num))
    return tc_list


def main(args):
    test_dict = init_test(args.device, args.brief_show, args.board, args.config_file)

    # shuffle test case
    if args.shuffle:
        for key in test_dict:
            test_dict[key].set_shuffle_test()

    # list all test suite
    if args.list_test:
        for key in test_dict:
            print(key)
    # show testcase
    elif args.show:
        key = args.show.lower()
        if key == 'all':
            for k in test_dict:
                print('[' + k + ' test]')
                test_dict[k].show_tc_list()
        elif key in test_dict:
            test_dict[key].show_tc_list()
        else:
            print("No such test : " + args.show)
            print("Please check test name with --list_test option")
    # run test
    elif args.run:
        key = args.run.lower()
        if key in test_dict:
            test_dict[key].run_test(0)
        else:
            print("No such test : " + args.show)
            print("Please check test name with --list_test option")
    # sqe test
    elif args.sqe_tc:
        test_dict['sqe'].run_test_list(make_list_from_str(args.sqe_tc))
    # e2e test
    elif args.e2e_tc:
        test_dict['e2e'].run_test_list(make_list_from_str(args.e2e_tc))
    #default
    else:
        print("Use --help option to show help message")

    # for report to CI
    ret = 0
    for key in test_dict:
        ret += test_dict[key].get_fail_cnt()
    sys.exit(ret)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='enn_test.py',
        epilog='https://github.sec.samsung.net/SWAT/UENN_TEST'
        )
    parser.add_argument('--list_test', action='store_true',
                        help='show all testsuite name')
    parser.add_argument('--show', type=str, default='',
                        help='show test cases : all, e2e, sqe')
    parser.add_argument('--run', type=str, default='',
                        help='run all test cases in test : e2e, sqe')
    parser.add_argument('--e2e_tc', type=str, default='',
                        help='End to End test case number or list (e.g. --e2e_tc 1 / --e2e_tc [1,1,3])')
    parser.add_argument('--sqe_tc', type=str, default='',
                        help='SQE test case number or list (e.g. --sqe_tc 1 / --sqe_tc [1,1,3])')
    parser.add_argument('--device', type=str, default='',
                        help='adb device id')
    parser.add_argument('--brief', dest='brief_show', action='store_true',
                        help='Print only results briefly')
    parser.set_defaults(brief_show=False)
    parser.add_argument('--shuffle', action='store_true',
                        help='shuffle test cases')
    parser.add_argument('--board', type=str, default='',
                        help='Board name (e.g. exynos9925)')
    parser.add_argument('--config_file', type=str, default='',
                        help='Name of yaml config file')
    args = parser.parse_args()
    main(args)
