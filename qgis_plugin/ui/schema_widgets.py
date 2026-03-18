"""Build Qt widgets dynamically from JSON Schema parameter definitions."""

from collections import OrderedDict

from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QSpinBox,
    QWidget,
)


def build_param_widgets(schema: dict) -> OrderedDict:
    """Create Qt widgets for each property in a JSON Schema.

    Returns:
        OrderedDict mapping parameter name to (label, widget).
    """
    widgets = OrderedDict()
    properties = schema.get("properties", {})

    for name, prop in properties.items():
        ptype = prop.get("type", "string")
        default = prop.get("default")
        description = prop.get("description", name)
        label = _name_to_label(name)

        if ptype == "integer":
            w = QSpinBox()
            w.setMinimum(prop.get("minimum", 0))
            w.setMaximum(prop.get("maximum", 999999))
            if default is not None:
                w.setValue(int(default))
            w.setToolTip(description)

        elif ptype == "number":
            w = QDoubleSpinBox()
            w.setDecimals(4)
            w.setMinimum(prop.get("minimum", 0.0))
            w.setMaximum(prop.get("maximum", 999999.0))
            w.setSingleStep(0.1)
            if default is not None:
                w.setValue(float(default))
            w.setToolTip(description)

        elif ptype == "boolean":
            w = QCheckBox()
            if default is not None:
                w.setChecked(bool(default))
            w.setToolTip(description)

        elif ptype == "array":
            w = QLineEdit()
            if default is not None:
                w.setText(", ".join(str(v) for v in default))
            w.setPlaceholderText("e.g. 0, 1, 2")
            w.setToolTip(description)

        else:
            # Fallback: string/mixed type (e.g. dist_thres with 'auto')
            w = QLineEdit()
            if default is not None:
                w.setText(str(default))
            w.setToolTip(description)

        widgets[name] = (label, w)

    return widgets


def collect_param_values(widgets: OrderedDict) -> dict:
    """Read current values from widgets built by build_param_widgets."""
    values = {}
    for name, (_label, w) in widgets.items():
        if isinstance(w, QSpinBox):
            values[name] = w.value()
        elif isinstance(w, QDoubleSpinBox):
            values[name] = w.value()
        elif isinstance(w, QCheckBox):
            values[name] = w.isChecked()
        elif isinstance(w, QLineEdit):
            text = w.text().strip()
            # Try to parse as number, then comma-separated list, else string
            if "," in text:
                try:
                    values[name] = [int(v.strip()) for v in text.split(",")]
                except ValueError:
                    values[name] = text
            else:
                try:
                    values[name] = int(text)
                except ValueError:
                    try:
                        values[name] = float(text)
                    except ValueError:
                        values[name] = text if text else None
    return values


def create_param_group(title: str, widgets: OrderedDict) -> QGroupBox:
    """Wrap widgets in a QGroupBox with a QFormLayout."""
    group = QGroupBox(title)
    layout = QFormLayout()
    for _name, (label, widget) in widgets.items():
        layout.addRow(label, widget)
    group.setLayout(layout)
    return group


def _name_to_label(name: str) -> str:
    """Convert snake_case parameter name to a readable label."""
    return name.replace("_", " ").title()
