import re

# Unit: one or more digits, then one or more non [digit, '-', '_', space]
U = r"\d+[^\d\-_ ]+"

# Group: one or more units separated by underscores
GRP = rf"{U}(?:_{U})*"

# Word: u - grp - grp
W = rf"{U}-{GRP}-{GRP}"

# Final: one or more words separated by single spaces
PATTERN = re.compile(rf"^{W}(?: {W})*$")


def matches_signature_pattern(s: str) -> bool:
    """
    Return True iff `s` matches:
        word (space word)*
    where a word is:
        u - grp - grp
    u   = digits+ then chars+ (chars exclude digits, '-', '_', and space)
    grp = one or more u separated by '_'
    """
    return bool(PATTERN.fullmatch(s))


def matches_unit_pattern(s: str) -> bool:
    """Return True iff `s` matches the unit pattern."""
    return bool(re.fullmatch(U, s))


def matches_group_pattern(s: str) -> bool:
    """Return True iff `s` matches the group pattern."""
    return bool(re.fullmatch(GRP, s))


def matches_word_pattern(s: str) -> bool:
    """Return True iff `s` matches the word pattern."""
    return bool(re.fullmatch(W, s))
