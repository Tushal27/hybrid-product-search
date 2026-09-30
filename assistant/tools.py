"""
Real tool implementations for the personal assistant. Every tool here is
deterministic, offline, and safe to call -- no eval() on arbitrary text,
no network dependency.
"""

import ast
import datetime
import operator

# ---------------------------------- calculator (safe -- no eval()) ----------------------------------
_ALLOWED_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression element: {ast.dump(node)}")


def calculator(expression):
    """Safely evaluate a pure arithmetic expression -- +, -, *, /, **, %, //, parens.
    Rejects anything else (names, calls, attribute access) instead of using eval()."""
    try:
        tree = ast.parse(expression, mode="eval")
        return _eval_node(tree.body)
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError) as e:
        raise ValueError(f"could not evaluate {expression!r}: {e}")


# ---------------------------------- unit conversion ----------------------------------
_LENGTH_TO_METERS = {"m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001,
                      "mi": 1609.34, "ft": 0.3048, "in": 0.0254}
_WEIGHT_TO_GRAMS = {"g": 1.0, "kg": 1000.0, "mg": 0.001, "lb": 453.592, "oz": 28.3495}

# a model asked to name a unit is at least as likely to say "kilometers" as "km" --
# normalize common full names/plurals instead of requiring the exact abbreviation
_UNIT_ALIASES = {
    "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "kilometer": "km", "kilometers": "km", "kilometre": "km", "kilometres": "km",
    "centimeter": "cm", "centimeters": "cm", "millimeter": "mm", "millimeters": "mm",
    "mile": "mi", "miles": "mi", "foot": "ft", "feet": "ft", "inch": "in", "inches": "in",
    "gram": "g", "grams": "g", "kilogram": "kg", "kilograms": "kg", "milligram": "mg", "milligrams": "mg",
    "pound": "lb", "pounds": "lb", "ounce": "oz", "ounces": "oz",
    "celsius": "c", "fahrenheit": "f", "kelvin": "k",
}


def convert_units(value, from_unit, to_unit):
    from_unit = _UNIT_ALIASES.get(from_unit.lower(), from_unit.lower())
    to_unit = _UNIT_ALIASES.get(to_unit.lower(), to_unit.lower())
    if from_unit in _LENGTH_TO_METERS and to_unit in _LENGTH_TO_METERS:
        return value * _LENGTH_TO_METERS[from_unit] / _LENGTH_TO_METERS[to_unit]
    if from_unit in _WEIGHT_TO_GRAMS and to_unit in _WEIGHT_TO_GRAMS:
        return value * _WEIGHT_TO_GRAMS[from_unit] / _WEIGHT_TO_GRAMS[to_unit]
    if {from_unit, to_unit} <= {"c", "f", "k"}:
        celsius = {"c": value, "f": (value - 32) * 5 / 9, "k": value - 273.15}[from_unit]
        return {"c": celsius, "f": celsius * 9 / 5 + 32, "k": celsius + 273.15}[to_unit]
    raise ValueError(f"cannot convert between {from_unit!r} and {to_unit!r} "
                      f"(supported: length, weight, temperature -- not mixed categories)")


# ---------------------------------- date/time ----------------------------------
def get_datetime():
    return datetime.datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")


TOOL_SCHEMAS = {
    "calculator": {
        "description": "Evaluate an arithmetic expression and return the exact result",
        "inputSchema": {"type": "object", "properties": {"expression": {"type": "string"}},
                         "required": ["expression"]},
    },
    "convert_units": {
        "description": "Convert a value between units of length, weight, or temperature ONLY "
                        "(e.g. miles to km, lb to kg, F to C). Do NOT use this for percentages "
                        "or plain arithmetic -- use calculator for those.",
        "inputSchema": {"type": "object",
                         "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"},
                                        "to_unit": {"type": "string"}},
                         "required": ["value", "from_unit", "to_unit"]},
    },
    "get_datetime": {
        "description": "Get the current date and time",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
}
