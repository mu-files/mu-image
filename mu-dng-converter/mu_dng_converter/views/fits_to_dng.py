"""FITS → DNG conversion view."""

import flet as ft


def build_fits_view(page: ft.Page) -> ft.Control:
    """Build the FITS → DNG conversion tab content."""

    return ft.Column(
        controls=[
            ft.Text("FITS → DNG", size=16, weight=ft.FontWeight.BOLD),
            ft.Text("Coming soon.", size=13),
        ],
        expand=True,
    )
