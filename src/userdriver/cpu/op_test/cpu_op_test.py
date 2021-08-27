"""!@brief TestScript to manage(build/push/run) operators
"""
import argparse
import subprocess
import sys
import time
import glob
from colorama import Fore, Back, Style
# pylint: disable=W0622 (redefined-builtin)
from parse import compile


class TestUtils:
    """!@brief Utilities for TestScript
    """
    @staticmethod
    def alert_message_exit(msg, halt = True):
        """!@brief print alert message and exit
        @param msg log message
        """
        print('\033[31m' + msg + '\033[0m\n')
        if halt:
            sys.exit(0)

    @staticmethod
    def log_print(msg):
        """!@brief print log message
        @param msg log message
        """
        print("=" * 40)
        print("===== " + msg + " =====")
        print("=" * 40)

    @staticmethod
    def log_print_g(msg):
        """!@brief print log message with color
        @param msg log message
        """
        print("=" * 80)
        print(Fore.GREEN + Back.BLACK + Style.BRIGHT + msg + Fore.RESET + Back.RESET + Style.RESET_ALL)
        print("=" * 80)
        time.sleep(2)

    @staticmethod
    def shell(cmd):
        """!@brief perform bash cmd
        """
        process = subprocess.run(cmd, executable="/bin/bash", shell=True)
        if process.returncode != 0:
            TestUtils.alert_message_exit("Aborted command : {}".format(cmd), halt = False)

    @staticmethod
    def shell_return(cmd):
        """!@brief perform bash cmd and return returncode
        """
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
        return process.returncode

    @staticmethod
    def shell_with_output(cmd):
        """!@brief return stdout from bash cmd
        """
        result = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdoutdata, unused_stderrdata = result.communicate()
        return stdoutdata


class TestManager:
    """!@brief Manager of TestScript
    """

    _COMMAND_LIST = {
        'build': ['b'],
        'push': ['p'],
        'run': ['r'],
        'lcov': ['c'],
    }

    _TARGET_LIST = {
        'emulator': ['emul'],
        'universal2100': ['2100'],
    }

    _OPERATOR_LIST = list()

    _ROOT_DIR = '../../../../../../../../../..'

    _ANDROID_BP = '''cc_binary {{
    name: "{0}_test",
    vendor: true,
    rtti: true,
    defaults: [
        "enn_defaults",
    ],
    srcs: [
        "{0}_test.cpp",
    ],
    static_libs: [
        "libgtest",
        "libgtest_main",
    ],
    shared_libs: [
        "libenn_user_driver_cpu",
    ],
}}\n
'''

    _CMAKELISTS_TXT = '''set(SOURCE_FILES {0}_test.cpp ../operators/{1}.cpp)
add_executable({0}_test ${{SOURCE_FILES}})
target_link_libraries({0}_test ${{LIBRARY_FILES}})
add_test(NAME {0}_test COMMAND {0}_test)\n
'''

    _LCOV = 'lcov --capture --directory build/CMakeFiles/{0}.dir/ --output-file coverage/{0}_coverage.info'
    _GEN_HTML = 'genhtml coverage/{0}_coverage.info --output-directory coverage/{0}_coverage'

    _LCOV_FILTER = "'/usr/include/*' '*/userdriver/common/*' '*/userdriver/cpu/op_test/*' \
                    '*/userdriver/cpu/operators/*.hpp' '*/userdriver/cpu/operators/**/*.hpp'"
    _LCOV_INTEG_RAW = "lcov --capture --directory build/CMakeFiles/ -o coverage/cpu_op_test_coverage.raw"
    _LCOV_INTEG = "lcov -r coverage/cpu_op_test_coverage.raw {} -o coverage/cpu_op_test_coverage.info".format(_LCOV_FILTER)
    _GEN_HTML_INTEG = "genhtml coverage/cpu_op_test_coverage.info --output-directory coverage/cpu_op_test_coverage"

    def __init__(self):
        # Collect test list
        temp_op_list = glob.glob('*_test.cpp')
        for top in temp_op_list:
            self._OPERATOR_LIST.append(top.replace('test/', '').replace('_test.cpp', ''))
        self._OPERATOR_LIST.sort()

    def parse_args(self):
        """!@brief return parsed arguments
        """
        parser = argparse.ArgumentParser(description='\033[31m** CPU userdriver operator UnitTest Script **\033[0m')
        parser.add_argument('--command', '-c', dest='command', default='run',
                            help='select command {}'.format(TestManager._COMMAND_LIST.keys()))
        parser.add_argument('--target', '-t', dest='target', default='universal2100',
                            help='select target {}'.format(TestManager._TARGET_LIST.keys()))
        parser.add_argument('operator', nargs='*',
                            help='select operator {}'.format(self._OPERATOR_LIST))
        args = parser.parse_args()

        # Check invalid command
        if args.command not in TestManager._COMMAND_LIST.keys():
            TestUtils.alert_message_exit('Invalid command : {}'.format(args.command))

        # Check invalid target
        if args.target not in TestManager._TARGET_LIST.keys():
            TestUtils.alert_message_exit('Invalid target : {}'.format(args.target))

        # Check invalid test name
        for operator in args.operator:
            if operator == 'all':
                args.operator = self._OPERATOR_LIST
                break
            if operator not in self._OPERATOR_LIST:
                TestUtils.alert_message_exit('Invalid operator : {}'.format(operator))

        return args

    def process_build(self, args):
        """!@brief perform build CPU operator unittest
        """
        if args.target == 'emulator':
            # Generate CMakeLists.txt with test list
            if args.operator:
                print('Build operator :', args.operator)
                with open('CMakeLists.base') as base_file:
                    with open('CMakeLists.txt', 'w+') as txt_file:
                        txt_file.write(base_file.read())
                        for operator in args.operator:
                            class_name = self.get_class_name(operator)
                            if class_name:
                                txt_file.write(TestManager._CMAKELISTS_TXT.format(operator, class_name))

            # Emulator build for linux
            TestUtils.shell('mkdir build;\
                             cd build;\
                             cmake .. -DUNIT_TEST=true;\
                             make -j32;\
                             cd -')
        else:
            # Generate Android.bp with test list
            if args.operator:
                print('Build operator :', args.operator)
                with open('Android.bp.enn', 'w+') as bp_file:
                    for operator in args.operator:
                        bp_file.write(TestManager._ANDROID_BP.format(operator))

            # Run build
            TestUtils.shell('source {0}/build/envsetup.sh;\
                             lunch full_{1}_r-eng;\
                             cd ../../..;\
                             mm -j16;\
                             cd -'.format(TestManager._ROOT_DIR, args.target))

    # pylint: disable=R0201 (no-self-use)
    def process_push(self, args):
        """!@brief perform push CPU operator unittest file and library
        """
        if not args.operator:
            TestUtils.alert_message_exit('Please input operator to push in device')

        if args.target == 'emulator':
            print('\033[31mEmulator target do not push any files!!!\033[0m')
        else:
            cmd = 'cd {0}/out/target/product/{1}/vendor/; adb root; adb remount;\
                adb push lib /vendor/; adb push lib64 /vendor/;'.format(TestManager._ROOT_DIR, args.target)
            print('Push operator :', args.operator)
            for operator in args.operator:
                cmd += 'adb push bin/{}_test /vendor/bin/;'.format(operator)
            cmd += 'cd -'
            TestUtils.shell(cmd)

    # pylint: disable=R0201 (no-self-use)
    def process_run_test(self, args):
        """!@brief perform execute CPU operator unittest
        """
        if args.target == 'emulator':
            TestUtils.shell('cd build;\
                             ctest --output-on-failure;\
                             cd -')
        else:
            if not args.operator:
                TestUtils.alert_message_exit('Please input operator to run test')

            print('Test operator :', args.operator)
            for operator in args.operator:
                print('\n\033[33m************ Run {}_test ************\033[0m'.format(operator))
                TestUtils.shell('adb shell {}_test;'.format(operator))

    # pylint: disable=R0201 (no-self-use)
    def get_class_name(self, operator):
        """!@brief get operator class name from test file
        """
        with open('{0}_test.cpp'.format(operator)) as test_file:
            pattern = compile('#include "{path}/operators/{class}.hpp"\n')
            for line in test_file.readlines():
                parsed = pattern.parse(line)
                if parsed:
                    return parsed['class']
        return None

    def process_coverage_test(self, args):
        """!@brief perform execute CPU operator unittest
        """
        if args.target == 'emulator':
            self.process_run_test(args)
            TestUtils.shell('mkdir -p coverage')
            test_dir_list = glob.glob('build/CMakeFiles/*.dir')
            pattern = compile('build/CMakeFiles/{op_test}.dir')
            for test_dir in test_dir_list:
                test_name = pattern.parse(test_dir)['op_test']
                if test_name:
                    TestUtils.shell(TestManager._LCOV.format(test_name))
                    TestUtils.shell(TestManager._GEN_HTML.format(test_name))
        else:
            TestUtils.alert_message_exit('Not support yet for device!!!')

    def process_coverage_test_integ(self, args):
        """!@brief perform execute CPU operator unittest (integrated)
        """
        if args.target == 'emulator':
            self.process_run_test(args)
            TestUtils.shell('mkdir -p coverage')
            TestUtils.shell(TestManager._LCOV_INTEG_RAW)
            TestUtils.shell(TestManager._LCOV_INTEG)
            TestUtils.shell(TestManager._GEN_HTML_INTEG)
        else:
            TestUtils.alert_message_exit('Not support yet for device!!!')


def main():
    """!@brief Entrypoint
    """
    test_manager = TestManager()
    args = test_manager.parse_args()
    if args.command == 'build':
        test_manager.process_build(args)
    elif args.command == 'push':
        test_manager.process_push(args)
    elif args.command == 'lcov':
        test_manager.process_coverage_test_integ(args)
    else:
        test_manager.process_run_test(args)


if __name__ == '__main__':
    main()
