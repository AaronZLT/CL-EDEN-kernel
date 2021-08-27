import os
os.system("color")

def _get_essential_option():
    return os.environ.get('enn_test_brief_show', "Show All") != str(True)

def _print_green(string: str, next_line = '\n', prefix = ' #'):
    print(prefix + ' \033[32m' + string + '\033[0m', end=next_line, flush=True)

def _print_red(string: str, next_line = '\n', prefix = ' #'):
     print(prefix + ' \033[31m' + string + '\033[0m', end=next_line, flush=True)

def _print(str, next_line = '\n', prefix = ' #  [DBG] '):
    if _get_essential_option():
        print(prefix + str, end=next_line, flush=True)
