import sys
import json


def _parse_value(v):
    try:
        if "{" in v and "}" in v:  # parse dictionary
            try:
                return json.loads(v.replace("'", '"'))
            except Exception as e:
                pass

        if v.lower() == "true":
            return True
        elif v.lower() == "false":
            return False

        else:  # numeric
            return float(v) if "." in v else int(v)
    except:
        return v  # default str


def parse_args(args=sys.argv):
    args_updated = []
    for arg in args:
        args_updated.extend(arg.split("="))
    args = args_updated

    parsed_args = {}
    arg_no = 1
    while arg_no < len(args):
        arg = args[arg_no]
        if arg.startswith("-"):
            arg_name = arg.strip("-")
            arg_value = _parse_value(args[arg_no + 1])
            parsed_args[arg_name] = arg_value
            arg_no += 1
        arg_no += 1
    return parsed_args
