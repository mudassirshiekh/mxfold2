from cmake_build_extension import CMakeExtension, BuildExtension
from pathlib import Path

def build(setup_kwargs):
    ext_modules = [
        CMakeExtension(
            name="mxfold2.interface",
            install_prefix="mxfold2",
            source_dir=str(Path(__file__).parent.joinpath("mxfold2", "src")),
            cmake_depends_on=["pybind11"],
        ),
    ]
    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
            "cmdclass": dict(build_ext=BuildExtension),
            "zip_safe": False,
        }
    )
