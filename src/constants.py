from dotmap import DotMap

# Filenames
TEMPLATE_FILENAME = "template.json"
EVALUATION_FILENAME = "evaluation.json"
CONFIG_FILENAME = "config.json"

FIELD_LABEL_NUMBER_REGEX = r"([^\d]+)(\d*)"
#
ERROR_CODES = DotMap(
    {
        "MULTI_BUBBLE_WARN": 1,
        "NO_MARKER_ERR": 2,
    },
    _dynamic=False,
)

FIELD_TYPES = {
    "QTYPE_INT": {
        "bubbleValues": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        "direction": "vertical",
    },
    "TRUEFALSE": {"bubbleValues": [0, 1], "direction": "horizontal"},
    "QTYPE_MCQ3": {"bubbleValues": [0, 1, 2], "direction": "horizontal"},
    "QTYPE_MCQ4": {"bubbleValues": [0, 1, 2, 3], "direction": "horizontal"},
    "FORM": {
        "bubbleValues": [0, 1, 2, 3],
        "direction": "vertical",
    },
    #
    # You can create and append custom field types here-
    #
}

TEXT_SIZE = 0.95
CLR_BLACK = (50, 150, 150)
CLR_WHITE = (250, 250, 250)
CLR_GRAY = (130, 130, 130)
CLR_DARK_GRAY = (100, 100, 100)

GLOBAL_PAGE_THRESHOLD_WHITE = 200
GLOBAL_PAGE_THRESHOLD_BLACK = 100
