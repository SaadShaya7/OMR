import json
import re
from copy import deepcopy
from fractions import Fraction

from deepmerge import Merger

from defaults import TEMPLATE_DEFAULTS
from schemas.constants import FIELD_STRING_REGEX_GROUPS
import logger


def load_json(path, **rest):
    try:
        with open(path, "r") as f:
            loaded = json.load(f, **rest)
    except json.decoder.JSONDecodeError as error:
        logger.critical(f"Error when loading json file at: '{path}'\n{error}")
        exit(1)
    return loaded


OVERRIDE_MERGER = Merger(
    # pass in a list of tuples,with the
    # strategies you are looking to apply
    # to each type.
    [
        # (list, ["prepend"]),
        (dict, ["merge"])
    ],
    # next, choose the fallback strategies,
    # applied to all other types:
    ["override"],
    # finally, choose the strategies in
    # the case where the types conflict:
    ["override"],
)


def open_template_with_defaults(template_path):
    user_template = load_json(template_path)
    user_template = OVERRIDE_MERGER.merge(deepcopy(TEMPLATE_DEFAULTS), user_template)
    return user_template


def parse_fields(key, fields):
    parsed_fields = []
    fields_set = set()
    for field_string in fields:
        fields_array = parse_field_string(field_string)
        current_set = set(fields_array)
        if not fields_set.isdisjoint(current_set):
            raise Exception(
                f"Given field string '{field_string}' has overlapping field(s) with other fields in '{key}': {fields}"
            )
        fields_set.update(current_set)
        parsed_fields.extend(fields_array)
    return parsed_fields


def parse_field_string(field_string):
    if "." in field_string:
        field_prefix, start, end = re.findall(FIELD_STRING_REGEX_GROUPS, field_string)[
            0
        ]
        start, end = int(start), int(end)
        if start >= end:
            raise Exception(
                f"Invalid range in fields string: '{field_string}', start: {start} is not less than end: {end}"
            )
        return [
            f"{field_prefix}{field_number}" for field_number in range(start, end + 1)
        ]
    else:
        return [field_string]


def parse_float_or_fraction(result):
    if type(result) == str and "/" in result:
        result = float(Fraction(result))
    else:
        result = float(result)
    return result
