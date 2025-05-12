def generate_fieldnames(keys, suffixes, prefix="diag"):
    """
    Generate all diag/<key>/<suffix> fieldnames for given keys and suffixes.
    """
    fieldnames = []
    for key in keys:
        for suffix in suffixes:
            fieldnames.append(f"{prefix}/{key}/{suffix}")
    return fieldnames