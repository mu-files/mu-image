"""mu dng converter - main application entry point."""

import flet as ft

from .views.dng_convert import build_dng_view
from .views.fits_to_dng import build_fits_view


def app(page: ft.Page):
    """Main Flet application."""
    page.title = "mu DNG Converter"
    page.window.width = 720
    page.window.height = 780
    page.padding = 20
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#3a3a3a"
    page.window.bgcolor = "#3a3a3a"
    page.window.icon = "icon.png"
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

    dng_dir_picker = ft.FilePicker()
    dng_file_picker = ft.FilePicker()
    dng_save_picker = ft.FilePicker()
    fits_dir_picker = ft.FilePicker()
    fits_file_picker = ft.FilePicker()

    tabs = ft.Tabs(
        selected_index=0,
        length=2,
        expand=True,
        content=ft.Column(
            expand=True,
            controls=[
                ft.TabBar(
                    secondary=True,
                    tabs=[
                        ft.Tab(label="DNG \u2192 Image"),
                        ft.Tab(label="FITS \u2192 DNG"),
                    ],
                ),
                ft.TabBarView(
                    expand=True,
                    controls=[
                        ft.Container(
                            content=build_dng_view(page,
                                dir_picker=dng_dir_picker,
                                file_picker=dng_file_picker,
                                save_picker=dng_save_picker),
                            padding=ft.Padding(
                                left=0, top=10, right=0, bottom=0
                            ),
                            expand=True,
                        ),
                        ft.Container(
                            content=build_fits_view(page,
                                dir_picker=fits_dir_picker,
                                file_picker=fits_file_picker),
                            padding=ft.Padding(
                                left=0, top=10, right=0, bottom=0
                            ),
                            expand=True,
                        ),
                    ],
                ),
            ],
        ),
    )

    page.add(
        ft.SafeArea(
            expand=True,
            content=tabs,
        )
    )


def main():
    """Entry point for CLI."""
    ft.run(app, assets_dir="assets")


if __name__ == "__main__":
    main()
