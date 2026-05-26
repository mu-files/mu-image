"""mu dng converter - main application entry point."""

import flet as ft

from .views.dng_convert import build_dng_view


def app(page: ft.Page):
    """Main Flet application."""
    page.title = "mu DNG Converter"
    page.window.width = 720
    page.window.height = 780
    page.padding = 20
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#3a3a3a"
    page.window.bgcolor = "#3a3a3a"
    page.window.icon = "muIcon_muimg.png"
    page.dark_theme = ft.Theme(
        color_scheme=ft.ColorScheme(
            surface="#3a3a3a",
            surface_container="#434343",
            surface_container_low="#3a3a3a",
            surface_container_lowest="#323232",
            surface_container_high="#4a4a4a",
            surface_container_highest="#535353",
            on_surface="#e0e0e0",
            on_surface_variant="#b0b0b0",
            primary="#6bb5ff",
        ),
    )

    page.add(
        ft.SafeArea(
            expand=True,
            content=build_dng_view(page),
        )
    )


def main():
    """Entry point for CLI."""
    ft.run(app, assets_dir="assets")


if __name__ == "__main__":
    main()
