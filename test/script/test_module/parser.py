# -*- coding: utf-8 -*-
#!/usr/bin/env python
import os
import yaml
from abc import *

cmd_frame = {'run' : '{cmd}',
             'open_close' : 'for i in \`seq {cnt}\`; do {cmd}; sleep 1; done',
             'multi_stress' : '({cmd1} &); ({cmd2} &)',
             'anr' : '{cmd1}; for i in \`seq {cnt}\`; do timeout 5s {cmd2}; done',
             'stress_anr' : '({cmd1} &); ({cmd2} &); for i in \`seq {cnt}\`; do timeout 9s {cmd3}; done'}


class Testcase(metaclass = ABCMeta):
    def generate_command(self, vector, report_file, iter, repeat, is_block, is_lib, mode=''):
        test_cmd = 'taskset 70'
        if is_lib:
            test_cmd += ' EnnTest_v2_lib'
        else:
            test_cmd += ' EnnTest_v2_service'
        test_cmd += ' --model ' + vector['model']
        test_cmd += ' --input'
        for input in vector['input']:
            test_cmd += ' ' + input
        test_cmd += ' --golden'
        for golden in vector['golden']:
            test_cmd += ' ' + golden
        test_cmd += ' --threshold ' + str(vector['threshold'])
        test_cmd += ' --iter ' + str(iter)
        test_cmd += ' --repeat ' + str(repeat)
        if not is_block:
            test_cmd += ' --async'
        test_cmd += ' --report ' + report_file
        if mode:
            test_cmd += ' --mode ' + mode

        return test_cmd

    @abstractmethod
    def get_description(self):
        pass

    @abstractmethod
    def get_check_memleak(self):
        pass

    @abstractmethod
    def get_command(self):
        pass


class SQETestcase(Testcase):
    def __init__(self, tc_dict, vector_list):
        self.tc_id = tc_dict['tc_id']
        self.type = tc_dict['type']
        self.description = tc_dict['description']
        self.check_memleak = tc_dict['check_memleak']
        self.subcmds_info = tc_dict['subcommands']
        self.subcmd_list = []
        self.command = ''
        self.update_command(vector_list)

    def update_subcmd_list(self, vector_list):
        for subcmd in self.subcmds_info:
            subcmd_dict = {}
            report_file = 'sqe_' + str(self.tc_id) + '_' + str(len(self.subcmd_list))
            for vector in vector_list:
                if subcmd['model'] in vector:
                    test_cmd = super().generate_command(vector[subcmd['model']], report_file, subcmd['iter'], \
                                                     subcmd['repeat'], subcmd['is_block'], subcmd['is_lib'])
                    break
            subcmd_dict['subcommand'] = test_cmd
            subcmd_dict['report_file'] = report_file
            subcmd_dict['is_anr'] = subcmd['is_anr']
            subcmd_dict['tc_cnt'] = subcmd['tc_cnt']
            self.subcmd_list.append(subcmd_dict)


    def update_command(self, vector_list):
        self.update_subcmd_list(vector_list)

        cmd_num = len(self.subcmd_list)
        if self.type == 'run' and cmd_num == 1:
            self.command = cmd_frame['run'].format(cmd=self.subcmd_list[0]['subcommand'])
        elif self.type == 'open_close' and cmd_num == 1:
            self.command = cmd_frame['open_close'].format(cmd=self.subcmd_list[0]['subcommand'], \
                                                  cnt=self.subcmd_list[0]['tc_cnt'])
        elif self.type == 'multi_stress' and cmd_num == 2:
            self.command = cmd_frame['multi_stress'].format(cmd1=self.subcmd_list[0]['subcommand'], \
                                                    cmd2=self.subcmd_list[1]['subcommand'])
        elif self.type == 'anr' and cmd_num == 2:
            self.command = cmd_frame['anr'].format(cmd1=self.subcmd_list[0]['subcommand'], \
                                           cmd2=self.subcmd_list[1]['subcommand'], \
                                           cnt=self.subcmd_list[1]['tc_cnt'])
        elif self.type == 'stress_anr' and cmd_num == 3:
            self.command = cmd_frame['stress_anr'].format(cmd1=self.subcmd_list[0]['subcommand'], \
                                                  cmd2=self.subcmd_list[1]['subcommand'], \
                                                  cmd3=self.subcmd_list[2]['subcommand'], \
                                                  cnt=self.subcmd_list[2]['tc_cnt'])
        else:
            self.command = 'Not Supported'

    def get_description(self):
        return self.description

    def get_check_memleak(self):
        return self.check_memleak

    def get_command(self):
        return self.command

    def get_subcmd_list(self):
        return self.subcmd_list


class ENNTestcase(Testcase):
    def __init__(self, tc_dict, vector_list):
        self.description = tc_dict['description']
        self.is_lib = tc_dict['is_lib']
        self.is_block = tc_dict['is_block']
        self.check_memleak = tc_dict['check_memleak']
        self.iter = tc_dict['iter']
        self.repeat = tc_dict['repeat']
        self.model_list = tc_dict['model']
        self.mode = tc_dict['mode']
        self.command_dict = {}
        self.update_command_dict(vector_list)

    def update_command_dict(self, vector_list):
        for model in self.model_list:
            report_file = self.description + '_' + model

            command = 'Not supported'
            for vector in vector_list:
                if model in vector:
                    command = super().generate_command(vector[model], report_file, self.iter, \
                                                       self.repeat, self.is_block, self.is_lib, self.mode)
                    break
            self.command_dict[model] = {'command' : command, 'report_file' : report_file}

    def get_description(self):
        return self.description

    def get_check_memleak(self):
        return self.check_memleak

    def get_command(self):
        return self.command_dict


def load_config_file(file_name: str):
    file = 'test_config/' + file_name
    with open(file) as f:
        contents = yaml.load(f, Loader=yaml.FullLoader)
        return contents


def parse_test_case(key: str, dir_name: str, file_name: str, tc_list):
    tc_config_file = os.path.join(dir_name, file_name)
    vector_config_file = os.path.join(dir_name, 'test_vector.yaml')

    test_config = load_config_file(tc_config_file)
    vector = load_config_file(vector_config_file)['test_vector']

    for i, tc in enumerate(test_config[key]):
        if key == 'sqe':
            testcase = SQETestcase(tc, vector)
        else:
            testcase = ENNTestcase(tc, vector)
        tc_list[i+1] = testcase
