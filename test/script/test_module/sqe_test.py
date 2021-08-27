# -*- coding: utf-8 -*-
#!/usr/bin/env python
import os
import random
from time import sleep
from . import testsuite
from . import parser
from . import list_test
from . import test_func
import test_module.print_util as print_util

class SQE_Test (testsuite.Testsuite):
    def __init__(self, adb_manager: test_func.ADBManager):
        self.tc_num = 0
        self.tc_list = {}
        self.tc_name = 'sqe'
        self.config_dir = 'exynos9925'
        self.config_file = 'sqe_test.yaml'
        self.test_util = test_func.TestUtil(adb_manager)
        self.fail_cnt = 0
        self.is_shuffle = False

    def set_config_file(self, board, config_file):
        if board:
            self.config_dir = board
        if config_file:
            self.config_file = config_file

    def set_shuffle_test(self):
        self.is_shuffle = True

    def update_tc_list(self):
        parser.parse_test_case(self.tc_name, self.config_dir, self.config_file, self.tc_list)
        self.tc_num = len(self.tc_list)

    def show_tc_list(self):
        list_test.show_sqe_test_list(self.tc_list)

    def run_test(self, tc: int):
        if tc == 0:
            range_list = list(range(1,self.tc_num + 1))
            if self.is_shuffle:
                random.shuffle(range_list)

            for i in range_list:
                self._run_sqe_tc(i, self.tc_list[i])
                #sleep(10)   # sleep for cooling device
        else:
            if tc > self.tc_num or tc < 0:
                print_util._print_red('Invalid tc number (tc : 1 ~ ' + str(self.tc_num) + ')')
                return
            self._run_sqe_tc(tc, self.tc_list[tc])

    def run_test_list(self, tc_num_list: list):
        if self.is_shuffle:
                random.shuffle(tc_num_list)

        for i in tc_num_list:
            if i > 0:
                self.run_test(i)

    def get_fail_cnt(self):
        return self.fail_cnt

    def _update_fail_cnt(self, test_result, memleak_result):
        if not test_result or not memleak_result:
            self.fail_cnt += 1

    def _run_sqe_tc(self, tc_id: int, tc: parser.SQETestcase):
        out_file = os.path.join(self.test_util.out_dir, self.tc_name + '_' + str(tc_id))
        mem_file_before = os.path.join(self.test_util.mem_dir, self.tc_name + '_' + str(tc_id) + '_before')
        mem_file_after = os.path.join(self.test_util.mem_dir, self.tc_name + '_' + str(tc_id) + '_after')
        test_case_string = '[TC {n}] {tc}'.format(n=tc_id, tc=tc.get_description())
        print_util._print(test_case_string + ' Start', True)

        for subcmd in tc.get_subcmd_list():
            self.test_util.remove_result_form_device(subcmd['report_file'])

        memory_info = self.test_util.prepare_memleak_test(tc.get_check_memleak(), mem_file_before)

        pid = self.test_util.get_service_pid()

        # run test
        self.test_util.run_test_command(tc.get_command(), out_file)

        # validate test result
        test_result = True
        for subcmd in tc.get_subcmd_list():
            test_result = self.test_util.validate_sqe_result(self.test_util.ret_dir, subcmd['report_file'],
                                                             subcmd['tc_cnt'],subcmd['is_anr'], out_file)
            if test_result == False:
                break

        memleak_result = self.test_util.validate_memleak_test(tc.get_check_memleak(), memory_info, mem_file_after)
        self._update_fail_cnt(test_result, memleak_result)
        self.test_util.print_test_result(test_result, memleak_result, test_case_string)

        # validate service
        self.test_util.check_invalid_service_by_pid(pid)