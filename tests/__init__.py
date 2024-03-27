import sys
import inspect
import traceback
import shutil
import os


def get_test_functions():
    return [
        (name, obj)
        for name, obj in inspect.getmembers(sys.modules["__main__"])
        if (inspect.isfunction(obj) and
        name.startswith("test") and
        obj.__module__ == "__main__")
    ]

def run_tests():
    test_functions = get_test_functions()
    passing_tests = []
    failing_tests = []
    assertion_errors = []

    if os.path.isdir(os.path.join('.', '.testing')):
        shutil.rmtree(os.path.join('.', '.testing'))

    print("\nRunning tests:")
    for (name, test_function) in test_functions:
        print("")
        print(name)
        
        try:
            test_function()
            passing_tests.append(name)
        
        except AssertionError as e:
            failing_tests.append(name)
            assertion_errors.append( (e, traceback.format_exc()) )

    if os.path.isdir(os.path.join('.', '.testing')):
        shutil.rmtree(os.path.join('.', '.testing'))
    
    print("\nTest report:")
    print(
        f"\t{len(passing_tests)} passed, {len(failing_tests) } failed"
    )

    print("\nFailing tests:")
    for test, error in zip(failing_tests, assertion_errors):
        print("")
        print(test)
        print(error[1])
        print(error[0])
    
    if failing_tests:
        sys.exit(1)