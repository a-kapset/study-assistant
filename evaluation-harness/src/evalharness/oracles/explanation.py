"""Exact rules for the explanation of a wrongly answered practice question, checked before any judge reads it.

The app asks its model to name an option as the word "option" and the option's id, e.g. "option b". An explanation
that never names a correct option that way has not told the learner the right answer in a form a program can find,
whatever else it says.
"""

import re
from collections.abc import Iterable


def options_not_named(text: str, option_ids: Iterable[str]) -> list[str]:
    """Return the ids, in sorted order, that ``text`` never names as "option <id>".

    The word "option" or "options" may be in any letter case and may be followed by up to four characters that
    are not letters or digits, such as a space, a quote, a bracket or markdown stars; the id itself must be
    written exactly and must not run on into a longer word. So "Option **b**" names b, and "option be" does not.
    """
    missing = []
    for option_id in sorted(set(option_ids)):
        pattern = rf"(?i:\boptions?)[^\w]{{1,4}}{re.escape(option_id)}(?!\w)"
        if re.search(pattern, text) is None:
            missing.append(option_id)
    return missing
