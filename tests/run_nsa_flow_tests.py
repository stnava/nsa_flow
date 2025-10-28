#!/usr/bin/env python3
"""
Run all functions in the nsa_flow package that start with 'test_'.

- Dynamically imports all submodules of nsa_flow
- Finds all callables whose names start with test_
- Runs each test safely and reports results
"""

import importlib
import pkgutil
import traceback
import time
import sys
import types

# === configurable ===
PACKAGE_NAME = "nsa_flow"


def discover_submodules(package_name):
    """Recursively discover all submodules in a package."""
    pkg = importlib.import_module(package_name)
    modules = [pkg]
    for loader, name, is_pkg in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        try:
            mod = importlib.import_module(name)
            modules.append(mod)
        except Exception as e:
            print(f"[WARN] Could not import {name}: {e}")
    return modules


def find_test_functions(module):
    """Return list of (name, callable) for all test_* functions in a module."""
    funcs = []
    for name, obj in vars(module).items():
        if callable(obj) and name.startswith("test_"):
            funcs.append((name, obj))
    return funcs


def run_test(name, func):
    """Execute a test_* function and return result dict."""
    start = time.time()
    try:
        result = func()
        elapsed = time.time() - start
        return {"name": name, "status": "PASS", "time": elapsed, "result": result}
    except Exception as e:
        elapsed = time.time() - start
        return {
            "name": name,
            "status": "FAIL",
            "time": elapsed,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def main():
    print("=" * 80)
    print(f"🔍 Discovering NSA-Flow test functions in package: {PACKAGE_NAME}")
    print("=" * 80)

    modules = discover_submodules(PACKAGE_NAME)
    all_tests = []
    for mod in modules:
        all_tests.extend(find_test_functions(mod))

    print(f"✅ Found {len(all_tests)} test functions.")
    print("-" * 80)

    passed, failed = [], []
    for name, func in all_tests:
        print(f"▶ Running {name}() ... ", end="")
        res = run_test(name, func)
        if res["status"] == "PASS":
            print(f"✅ PASS ({res['time']:.3f}s)")
            passed.append(res)
        else:
            print(f"❌ FAIL ({res['time']:.3f}s)")
            failed.append(res)
            print(res["traceback"])

    print("=" * 80)
    print(f"✅ Passed: {len(passed)}   ❌ Failed: {len(failed)}   Total: {len(all_tests)}")
    print("=" * 80)

    if failed:
        print("🔴 Some tests failed:")
        for f in failed:
            print(f" - {f['name']}: {f['error']}")
        sys.exit(1)
    else:
        print("🎉 All tests passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
