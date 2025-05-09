import os
import re

rule = "^(feat|fix|bugfix|docs|style|refactor|perf|test|chore)\(.*\): [A-Z].*"  # noqa

error_hint = """
Merge Request Title Validation Failed:

{mr_title}

The title does not comply with rule: {rule}

Required Format: <type>(<scope>): <description>

Examples:
feat(api): Implement new authentication module

Validation Checklist:
1. Valid type? (feat|fix|bugfix|docs|style|perf|refactor|test|chore)
2. Description starts with capital letter
3. Colon has spaces on both sides
4. Contains non-ASCII characters
"""  # noqa


def main():
    branch = os.environ.get("gitlabTargetBranch")

    if branch != "master":
        return

    merge_request_title = os.environ["gitlabMergeRequestTitle"]

    if len(merge_request_title) > 128:
        raise ValueError(
            "The maximum length of merge request title is 128, but get {} for merge request title: {}".format(  # noqa
                len(merge_request_title), merge_request_title
            )
        )

    if not re.match(rule, merge_request_title):
        raise ValueError(
            error_hint.format(mr_title=merge_request_title, rule=rule)
        )


if __name__ == "__main__":
    if os.environ.get("gitlabActionType", "") == "MERGE":
        main()
