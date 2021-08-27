#!/usr/bin/python3
import sys, argparse

### Parameters
# constants
IN_FILE_DEFAULT="../../src/medium/hidl/enn/1.0/types.hal"
OUT_FILE_DEFAULT="../../src/medium/types.hal.h"
HEADER_FILE_DEFAULT="types_hal_header.txt"
FOOTER_FILE_DEFAULT="types_hal_footer.txt"

PATTERNS=[
    [ "package vendor.samsung_slsi.hardware.enn@1.0;", "" ],
    [ "handle fd", "handle_x86 data;\n    handle_x86 *fd = &data" ],
    [ "handle hd", "handle_x86 data;\n    handle_x86 *hd = &data" ],
    [ "string", "std::string" ],
    [ "vec<", "std::vector<" ],
]

def convert(in_f, out_f, h_f, f_f, pattern):
    try:
        with open(out_f, 'w') as fd_out, open(f_f, 'r') as fd_ft, open(h_f, 'r') as fd_hd, open(in_f, 'r') as fd_in:
            ## Write header
            print("# Write header..")
            hd_lines = fd_hd.readlines()
            fd_out.write("//// This file is generates from " + in_f)
            fd_out.write("\r\n\r\n")
            for hd_line in hd_lines:
                fd_out.write(hd_line)

            ## Convert input file
            print("# Convert Input..")
            input_lines = fd_in.readlines()
            for input_line in input_lines:
                for pattern_element in pattern:
                    input_line = input_line.replace(pattern_element[0], pattern_element[1])
                fd_out.write(input_line)

            ## Write footer
            print("# Write footer..")
            ft_lines = fd_ft.readlines()
            fd_out.write("\r\n\r\n")
            for ft_line in ft_lines:
                fd_out.write(ft_line)
    except FileNotFoundError as e:
        print('#', e)
        exit(2)

def main(argv):
    # Create argument from argparse
    parser = argparse.ArgumentParser(description='Android hal converter to c++ header')

    # register parser
    parser.add_argument('--input', '-i', required=False, default=IN_FILE_DEFAULT, help='set input file, default: ' + IN_FILE_DEFAULT)
    parser.add_argument('--output', '-o', required=False, default=OUT_FILE_DEFAULT, help='set output file, default: ' + OUT_FILE_DEFAULT)
    parser.add_argument('--header', required=False, default=HEADER_FILE_DEFAULT, help='set header file, default: ' + HEADER_FILE_DEFAULT)
    parser.add_argument('--footer', required=False, default=FOOTER_FILE_DEFAULT, help='set footer file, default: ' + FOOTER_FILE_DEFAULT)

    args = parser.parse_args()
    for arg in vars(args):
        print("# Arguments:", arg, ":", getattr(args, arg))
    print("\r\n# Start to convert.. ")
    convert(args.input, args.output, args.header, args.footer, PATTERNS)
    print("# Finished to convert.. done \r\n")

if __name__ == "__main__":
    main(sys.argv)