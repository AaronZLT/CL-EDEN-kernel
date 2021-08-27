# -*- coding: utf-8 -*-
#!/usr/bin/env python
from abc import *

class Testsuite(metaclass = ABCMeta):
    @abstractmethod
    def set_shuffle_test(self):
        # shuffle test cases when this function called
        pass

    @abstractmethod
    def update_tc_list(self):
        # update test case list by using config file
        pass

    @abstractmethod
    def show_tc_list(self):
        # Show test case list by using texttable
        pass

    @abstractmethod
    def run_test(self, tc: int):
        # run test case with given tc number
        # run all test case if tc number is 0
        pass

    @abstractmethod
    def run_test_list(self, tc_num_list: list):
        # run test cases with given list of tc numbers
        pass

    @abstractmethod
    def get_fail_cnt(self):
        # return fail count to report to CI
        pass