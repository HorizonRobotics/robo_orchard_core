import argparse
import os
import subprocess


def get_root():
    current_file_path = os.path.abspath(__file__)
    root_path = os.path.dirname(current_file_path)
    for _ in range(2):
        root_path = os.path.dirname(root_path)
    return root_path


def python_lint(root_path: str, auto_format: bool = False):
    # run external python file to lint python

    current_file_path = os.path.abspath(__file__)
    cur_folder = os.path.dirname(current_file_path)

    subprocess.check_call(
        " ".join(
            [
                "bash",
                (f"{cur_folder}/code_lint_py.sh"),
                f"{root_path}/",
                "true" if auto_format else "false",
            ]
        ),
        shell=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="check format.")
    parser.add_argument(
        "--auto_format", action="store_true", help="auto format python"
    )
    parser = parser.parse_args()
    root_path = get_root()
    # cpp_lint(root_path) # not available yet
    python_lint(root_path, parser.auto_format)
