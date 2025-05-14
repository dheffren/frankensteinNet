def flatten(named_dict, prefix=""):
    for k, v in named_dict.items():
        key = f"{prefix}_{k}" if prefix else k
        if isinstance(v, dict):
            yield from flatten(v, key)
        else:
            yield key, v

def safe_keyname(keyname, rep = "_"):
    return keyname.replace("/", rep)