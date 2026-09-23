"""Scientific roles map to shared house colour names."""
from .panels import colour, grey

ROLES = {"reporter": "circadian_purple", "surveillance": "teal", "morphology": "orange",
         "motility": "blue", "passed": "circadian_green", "failed": "red", "outline":"white"}


def role(name):
    return colour(ROLES.get(name, name))
