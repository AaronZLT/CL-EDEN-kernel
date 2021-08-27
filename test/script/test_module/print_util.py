import os

def _get_essential_option():
    return os.environ.get('enn_test_brief_show', "Show All") != str(True)

def _print_green(string: str, is_forced = True):
    if _get_essential_option() or is_forced:
        print(' #  \033[32m' + string + '\033[0m')

def _print_red(string: str, is_forced = True):
    if _get_essential_option() or is_forced:
        print(' #  \033[31m' + string + '\033[0m')

def _print(string: str, is_forced = False):
    if _get_essential_option() or is_forced:
        print(' #  ' + string)
