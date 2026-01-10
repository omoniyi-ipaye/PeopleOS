"""
Design System for PeopleOS.

Clean & minimal design inspired by Notion and Linear.
Provides consistent design tokens for typography, colors, spacing, and shadows.
"""

from typing import Dict, Any

# Typography scale - Inter font family
TYPOGRAPHY = {
    "font_family": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    "font_import": "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');",
    "h1": {"size": "28px", "weight": 600, "line_height": 1.2},
    "h2": {"size": "22px", "weight": 600, "line_height": 1.3},
    "h3": {"size": "18px", "weight": 500, "line_height": 1.4},
    "h4": {"size": "16px", "weight": 500, "line_height": 1.4},
    "body": {"size": "14px", "weight": 400, "line_height": 1.5},
    "body_small": {"size": "13px", "weight": 400, "line_height": 1.5},
    "caption": {"size": "12px", "weight": 400, "line_height": 1.4},
    "label": {"size": "11px", "weight": 500, "line_height": 1.4, "letter_spacing": "0.5px"},
}

# Clean & minimal color palette
COLORS_LIGHT = {
    # Backgrounds
    "background": "#ffffff",
    "surface": "#f9fafb",
    "surface_hover": "#f3f4f6",
    "surface_active": "#e5e7eb",

    # Borders
    "border": "#e5e7eb",
    "border_subtle": "#f3f4f6",
    "border_focus": "#3b82f6",

    # Text
    "text_primary": "#111827",
    "text_secondary": "#6b7280",
    "text_tertiary": "#9ca3af",
    "text_inverse": "#ffffff",

    # Accent colors
    "accent": "#3b82f6",
    "accent_hover": "#2563eb",
    "accent_light": "#eff6ff",

    # Status colors
    "success": "#059669",
    "success_light": "#ecfdf5",
    "warning": "#d97706",
    "warning_light": "#fffbeb",
    "danger": "#dc2626",
    "danger_light": "#fef2f2",
    "info": "#0284c7",
    "info_light": "#f0f9ff",

    # Risk colors
    "risk_high": "#ef4444",
    "risk_medium": "#f59e0b",
    "risk_low": "#22c55e",

    # Chart colors
    "chart_palette": ["#3b82f6", "#8b5cf6", "#06b6d4", "#10b981", "#f59e0b", "#ef4444"],
}

COLORS_DARK = {
    # Backgrounds
    "background": "#0f172a",
    "surface": "#1e293b",
    "surface_hover": "#334155",
    "surface_active": "#475569",

    # Borders
    "border": "#334155",
    "border_subtle": "#1e293b",
    "border_focus": "#3b82f6",

    # Text
    "text_primary": "#f8fafc",
    "text_secondary": "#94a3b8",
    "text_tertiary": "#64748b",
    "text_inverse": "#0f172a",

    # Accent colors
    "accent": "#3b82f6",
    "accent_hover": "#60a5fa",
    "accent_light": "rgba(59, 130, 246, 0.15)",

    # Status colors
    "success": "#22c55e",
    "success_light": "rgba(34, 197, 94, 0.15)",
    "warning": "#f59e0b",
    "warning_light": "rgba(245, 158, 11, 0.15)",
    "danger": "#ef4444",
    "danger_light": "rgba(239, 68, 68, 0.15)",
    "info": "#0ea5e9",
    "info_light": "rgba(14, 165, 233, 0.15)",

    # Risk colors
    "risk_high": "#ef4444",
    "risk_medium": "#f59e0b",
    "risk_low": "#22c55e",

    # Chart colors
    "chart_palette": ["#60a5fa", "#a78bfa", "#22d3ee", "#34d399", "#fbbf24", "#f87171"],
}

# Spacing scale (8px base unit)
SPACING = {
    "0": "0px",
    "1": "4px",
    "2": "8px",
    "3": "12px",
    "4": "16px",
    "5": "20px",
    "6": "24px",
    "8": "32px",
    "10": "40px",
    "12": "48px",
    "16": "64px",
}

# Border radius scale
RADIUS = {
    "none": "0px",
    "sm": "4px",
    "md": "8px",
    "lg": "12px",
    "xl": "16px",
    "full": "9999px",
}

# Shadow scale (minimal, subtle shadows)
SHADOWS = {
    "none": "none",
    "xs": "0 1px 2px 0 rgba(0, 0, 0, 0.03)",
    "sm": "0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05)",
    "md": "0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05)",
    "lg": "0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -4px rgba(0, 0, 0, 0.05)",
    "xl": "0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05)",
}

# Transition timing
TRANSITIONS = {
    "fast": "150ms ease",
    "base": "200ms ease",
    "slow": "300ms ease",
}


def get_colors(dark_mode: bool = False) -> Dict[str, str]:
    """Get color palette for current theme."""
    return COLORS_DARK if dark_mode else COLORS_LIGHT


def get_design_tokens(dark_mode: bool = False) -> Dict[str, Any]:
    """Get all design tokens for current theme."""
    return {
        "colors": get_colors(dark_mode),
        "typography": TYPOGRAPHY,
        "spacing": SPACING,
        "radius": RADIUS,
        "shadows": SHADOWS,
        "transitions": TRANSITIONS,
    }


def generate_css_variables(dark_mode: bool = False) -> str:
    """Generate CSS custom properties from design tokens."""
    colors = get_colors(dark_mode)

    css_vars = [":root {"]

    # Color variables
    for name, value in colors.items():
        css_vars.append(f"  --color-{name.replace('_', '-')}: {value};")

    # Spacing variables
    for name, value in SPACING.items():
        css_vars.append(f"  --spacing-{name}: {value};")

    # Radius variables
    for name, value in RADIUS.items():
        css_vars.append(f"  --radius-{name}: {value};")

    # Shadow variables
    for name, value in SHADOWS.items():
        css_vars.append(f"  --shadow-{name}: {value};")

    css_vars.append("}")

    return "\n".join(css_vars)
