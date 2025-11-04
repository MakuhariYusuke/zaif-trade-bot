#!/usr/bin/env python3
"""
Test Results Summary Script
"""

import xml.etree.ElementTree as ET
import os

def main():
    # Parse test results
    if not os.path.exists('test-results.xml'):
        print("Test results file not found")
        return

    tree = ET.parse('test-results.xml')
    root = tree.getroot()

    # Extract test summary
    testsuites = root.findall('testsuite')
    total_tests = sum(int(ts.get('tests', 0)) for ts in testsuites)
    total_failures = sum(int(ts.get('failures', 0)) for ts in testsuites)
    total_errors = sum(int(ts.get('errors', 0)) for ts in testsuites)
    total_time = sum(float(ts.get('time', 0)) for ts in testsuites)

    print('=== Action Signal Guide Test Results Summary ===')
    print(f'Total Tests: {total_tests}')
    print(f'Passed: {total_tests - total_failures - total_errors}')
    print(f'Failed: {total_failures}')
    print(f'Errors: {total_errors}')
    print(f'Total Time: {total_time:.2f}s')
    print(f'Success Rate: {((total_tests - total_failures - total_errors) / total_tests * 100):.1f}%')

    # Component breakdown
    print('\n=== Component Breakdown ===')
    for testsuite in testsuites:
        name = testsuite.get('name', 'Unknown')
        tests = int(testsuite.get('tests', 0))
        failures = int(testsuite.get('failures', 0))
        errors = int(testsuite.get('errors', 0))
        passed = tests - failures - errors

        print(f'{name}: {passed}/{tests} passed')

if __name__ == "__main__":
    main()