# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import argparse
import os
import subprocess
import shutil


# PEP 257 -- Docstring Conventions
"""A python script to build, testing, and collecting test coverage.

   As a default, it builds in ../build and generates the coverage data in ../coverage
   If the build directory already exists, it skips the building step.
"""

def validate_build_paths(build_dir, cmake_path):
    """Validate the CMakeLists.txt path."""
    os.mkdir(build_dir)
    # Check the CMakeLists.txt is reachable from the args.build_dir
    # with the args.cmake_relative_path
    cmake_filepath = os.path.join(
        build_dir, cmake_path, 'CMakeLists.txt')
    if not os.path.exists(cmake_filepath):
        # remove the build_dir to keep invariant
        os.rmdir(build_dir)
        raise ValueError('{} is not reachable.'.format(cmake_filepath))


def build_ctest(build_dir, cmake_path, sanitizer):
    """Build the unit tests via cmake and run ctests."""
    validate_build_paths(build_dir, cmake_path)
    cwd = os.getcwd()  # store the cwd before change it to the build_dir
    os.chdir(build_dir)
    nproc = str(os.cpu_count())  # instead of $(nproc)
    build_flags =None;# '-DUNIT_TEST=True'
    if sanitizer:
        if sanitizer == 'address':
            print("Address Sanitizer is On")
            build_flags = '-DENABLE_ASAN=True'
        elif sanitizer == 'leak':
            print("Leak Sanitizer is On")
            build_flags = '-DENABLE_LSAN=True'
        elif sanitizer == 'thread':
            print("Thread Sanitizer is On")
            build_flags = '-DENABLE_TSAN=True'
        else:
            raise ValueError('{} is not sanitizer type.'.format(sanitizer))
    # cmake configure
    try:
        if build_flags:
            print(build_flags)
            subprocess.run(['cmake', cmake_path, '-DUNIT_TEST=True', build_flags], check=True)
        else:
            subprocess.run(['cmake', cmake_path, '-DUNIT_TEST=True'], check=True)
    except subprocess.CalledProcessError as e:
        print('ENN_ERROR: CMake Failed')
        raise
    # Run tests
    # Build, same as `make -j $(nproc)`
    #subprocess.run(['cmake', '--build', '.', '--config', 'Debug',
    #                '--', '-j', nproc], check=True)
    try:
        subprocess.run(['make', '-j'], check=True)
    except subprocess.CalledProcessError as e:
        print('ENN_ERROR: Build Failed')
        raise
    # Run tests
    try:
        subprocess.run(['ctest', '-j', 'nproc', '--output-on-failure'], check=True)
    except subprocess.CalledProcessError as e:
        print('ENN_ERROR: Unit Test Fail')
        raise
    os.chdir(cwd)  # restore the current working directory, keep the invariant


def capture_coverage(src_dir, dest_dir):
    """Gather coverage information with `gcov` and generate the coverage HTML page."""
    # get 'sample?' name from src_dir
    output_filename = 'coverage'
    output_filepath = os.path.join(dest_dir, output_filename+'.info')

    # Capture coverage information and write it to output_filepath
    try:
        subprocess.run(['lcov', '--capture', '--directory', src_dir,
                       '--output-file', output_filepath], check=True)
    except subprocess.CalledProcessError as e:
        print('ENN_ERROR: LCOV CAPTURE FAILED')
        raise
    # Remove unnecessary files
    try:
        subprocess.run(['lcov', '--remove', output_filepath, '/usr/*', '*/model/schema/*',
                       '--output-file', output_filepath], check=True)
    except subprocess.CalledProcessError as e:
        print('ENN_ERROR: LCOV REMOVE FAILED')
        raise
    html_dirpath = os.path.join(dest_dir, output_filename)

    # Generate Visual Coverage HTML page from lcov output file
    try:
        subprocess.run(['genhtml', output_filepath,
                       '--output-directory', html_dirpath], check=True)
    except subprocess.CalledProcessError as e:
        print('ENN_ERROR: GENHTML FAILED')
        raise


def validate_output_dir_args(output_dir):
    """Validate the output_dir argument."""
    if os.path.exists(output_dir):
        raise ValueError('The output_dir exists.')
    os.mkdir(args.output_dir)


def capture_coverage_all(args):
    """Make a proper coverage dir from the coverage list and call capture_coverage()."""
    validate_output_dir_args(args.output_dir)
    #for coverage_end_path in coverage_input_dirs:
    #    coverage_dir = os.path.join(args.build_dir, 'CMakeFiles', coverage_end_path)
    #    capture_coverage(coverage_dir, args.output_dir)
    coverage_dir = os.path.join(args.build_dir)
    capture_coverage(coverage_dir, args.output_dir)


def main(args):
    # Skip build if the build directory exists.
    if args.clean:
        shutil.rmtree(args.output_dir, ignore_errors=True)
        shutil.rmtree(args.build_dir, ignore_errors=True)
        return
    if not os.path.exists(args.build_dir):
        build_ctest(args.build_dir, args.cmake_relative_path, args.sanitizer)
    capture_coverage_all(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build_dir',
                        type=str, default='build',
                        help='a path to cmake build directory.')
    parser.add_argument('--cmake_relative_path',
                        type=str, default='..',
                        help='a path to CMakeLists.txt from the build directory.')
    parser.add_argument('--output_dir',
                        type=str, default='coverage',
                        help='a path of a directory to store coverage output data')
    parser.add_argument('--clean', '-c',
                        action='store_true',
                        default=False,
                        help='Clean default build and output directories: build_dir(' + parser.get_default('build_dir') + '), output(' + parser.get_default('output_dir') + ')')
    parser.add_argument('--sanitizer', type=str, default=None,
                        help='Enable Sanitizer: address, leak, or thread')
    args = parser.parse_args()

    main(args)