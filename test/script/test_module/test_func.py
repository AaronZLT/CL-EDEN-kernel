# -*- coding: utf-8 -*-
#!/usr/bin/env python
import subprocess
import re
import os
from contextlib import suppress
import test_module.print_util as print_util

class ADBManager():
    def __init__(self):
        self.device_id = ''

    def set_device_id(self, device_id: int):
        self.device_id = device_id

    def _get_adb_cmd_with_device_id(self, command: str):
        if self.device_id:
            adb_command ='adb -s ' + self.device_id + ' ' + command
        else:
            adb_command ='adb ' + command
        return adb_command

    def run_adb_cmd(self, command: str):
        adb_output = ''
        adb_command = self._get_adb_cmd_with_device_id(command)
        with suppress(subprocess.CalledProcessError):
            adb_output = subprocess.check_output(adb_command,
                                                 universal_newlines=True,
                                                 shell=True)
        return adb_output

    def run_test_cmd(self, command: str, out_file: str):
        test_command = self._get_adb_cmd_with_device_id('shell \"' + command + '\" > ' + out_file)
        print_util._print('Run ' + test_command)
        proc = subprocess.Popen(test_command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True,
                            shell=True)
        proc.wait()

    def get_pid(self, process_name: str):
        pid = ''
        adb_output = self.run_adb_cmd('shell \"ps -ef\"')
        for string in adb_output.split('\n'):
            if process_name in string:
                pid = re.findall('\d+', string)[0]
                break
        print_util._print('PID of ' + process_name + ' is: ' + pid)
        return pid

    def pull_file(self, src_path: str, file_name: str, dest_path: str):
        ret = True
        if src_path[-1] != '/':
            src_path += '/'
        self.run_adb_cmd('pull ' + src_path + file_name + ' ' + dest_path)

        if not os.path.exists(os.path.join(dest_path, file_name)):
            print_util._print('file ' + dest_path + file_name + ' not exists')
            ret = False
        return ret


class TestUtil():
    def __init__(self, adb_manager: ADBManager):
        self.adb = adb_manager
        self.service_process = 'vendor.samsung_slsi.hardware.enn'
        self.out_dir = os.path.join(os.getcwd(), 'result')
        self.ret_dir = os.path.join(self.out_dir, 'ret')
        self.mem_dir = os.path.join(self.out_dir, 'mem')
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.ret_dir, exist_ok=True)
        os.makedirs(self.mem_dir, exist_ok=True)


    def _check_dmabuf_cnt(self):
        dmabuf_cnt = 0
        pid = self.get_service_pid()
        if not pid:
            print_util._print_red('Service is not running')
            return 0

        adb_output = self.adb.run_adb_cmd('shell \"ls /proc/' + pid + '/fd/ -al\"')
        for string in adb_output.split('\n'):
            if 'dmabuf' in string:
                dmabuf_cnt += 1

        print_util._print('dmabuf_cnt: ' + str(dmabuf_cnt))
        return dmabuf_cnt


    def _get_mem_info(self):
        adb_output = self.adb.run_adb_cmd('shell \"cat /d/dma_buf/bufinfo | grep bytes\"')
        print_util._print('mem_info : ' + adb_output)
        return re.findall("\d+", adb_output)


    def _parse_single_result(self, fp):
        total_num = 0
        pass_num = 0
        fail_num = 0

        while True:
            line = fp.readline()
            if not line : break
            if 'Total' in line:
                total_num = int(''.join(list(filter(str.isdigit, line))))

            if 'Pass' in line:
                pass_num = int(''.join(list(filter(str.isdigit, line))))

            if 'Fail' in line:
                fail_num = int(''.join(list(filter(str.isdigit, line))))
                break
        return [total_num, pass_num, fail_num]


    def _is_test_pass_by_result_file(self, path: str, tc_cnt: int):
        results = []
        ret = True

        fp = open(path, 'r')
        for i in range(tc_cnt):
            results = self._parse_single_result(fp)
            if results[0] == 0:
                print_util._print_red("no test result for {0}/{1} tc".format(i, tc_cnt))
                ret = False
                break

            if (results[0] != results[1]) | (results[2] > 0):
                print_util._print_red("Test failed : [total : {0} / pass : {1} / fail : {2}]".format(results[0], results[1], results[2]))
                ret = False

        fp.close()
        return ret


    def _is_test_pass_by_stdout_file(self, out_file: str):
        # For ANR Test (Result file isn't created after ANR test)
        fail_strings = ['FAIL', 'fail', 'Fail', 'ERROR', 'error', 'Error']

        with open(out_file, 'r') as file:
            text = file.read()

            if not text:
                print_util._print_red('out file is empty')
                return False

            for fail_string in fail_strings:
                if fail_string in text:
                    return False
        return True


    def _dump_meminfo(self, out_file: str):
        self.adb.run_adb_cmd('shell \"dmabuf_dump -a\" > ' + out_file)
        self.adb.run_adb_cmd('shell \"cat /d/dma_buf/bufinfo\" >> ' + out_file)


    def prepare_memleak_test(self, check_memleak: bool, out_file: str):
        if not check_memleak:
            return []

        self._dump_meminfo(out_file)
        before_meminfo = self._get_mem_info()
        before_dma_cnt = self._check_dmabuf_cnt()
        return {'meminfo' : before_meminfo, 'dma_cnt' : before_dma_cnt}


    def validate_memleak_test(self, check_memleak: bool, memory_info: dict, out_file: str):
        ret = True

        if not check_memleak:
            return True

        self._dump_meminfo(out_file)
        if self._get_mem_info() != memory_info['meminfo']:
            print_util._print_red('FAIL: MEMORY LEAK (buf_info)')
            ret = False

        if self._check_dmabuf_cnt() != memory_info['dma_cnt']:
            print_util._print_red('FAIL: MEMORY LEAK (dma_cnt)')
            ret = False

        return ret


    def validate_result(self, path: str, result_file_name: str, tc_cnt: int = 1):
        result_full_path = os.path.join(path, result_file_name)
        if not self.adb.pull_file('/data/vendor/enn/results/', result_file_name, path):
            print_util._print_red('Output file is not exist : ' + result_full_path)
            return False

        if not self._is_test_pass_by_result_file(result_full_path, tc_cnt):
            print_util._print_red('Test Failed')
            return False
        return True


    def validate_sqe_result(self, path: str, result_file_name: str, tc_cnt: int, is_anr: bool, out_file: str):
        if is_anr:
            return self._is_test_pass_by_stdout_file(out_file)

        return self.validate_result(path, result_file_name, tc_cnt)


    def remove_result_form_device(self, file_name: str):
        self.adb.run_adb_cmd('shell \"rm /data/vendor/enn/results/' + file_name + '\" &> /dev/null')


    def run_test_command(self, command: str, out_file: str):
        self.adb.run_test_cmd(command, out_file)


    def print_test_result(self, test_result: bool, memleak_result: bool, string: str):
        if test_result and memleak_result:
            print_util._print_green(string + ' PASS')
        else:
            print_util._print_red(string + ' FAILED')


    def get_service_pid(self):
        return self.adb.get_pid(self.service_process)


    def check_invalid_service_by_pid(self, before_pid : int):
        after_pid = self.get_service_pid()

        if before_pid != after_pid:
            print_util._print_red("FAIL: Invalid service PID ({0}/{1})".format(before_pid, after_pid))
            self.adb.run_adb_cmd('shell \"reboot bootloader\"')