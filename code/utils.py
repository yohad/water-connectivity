"""
Utility functions.
"""


def get_name_suffix(p: float, m: float) -> str:
    """
    Get the name suffix for a given p and m.
    """
    return f"_p{float(p)}_m{float(m)}".replace(".", "_")
