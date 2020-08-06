""" Configuration file for punch.py """
__config_version__ = 1

GLOBALS = {"serializer": "{{major}}.{{minor}}.{{patch}}"}

FILES = ["setup.py"]

VERSION = ["major", "minor", "patch"]

VCS = {
    "name": "git",
    "commit_message": (
        "Version updated from {{ current_version }}" " to {{ new_version }}"
    ),
    "options": {
        "target_branch": "release",
        "make_release_branch": True,
        "annotate_tags": True,
    },
}
