import os
import shutil
from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name="local_package",
    version="1.0",
    packages=find_packages(),
    ext_modules=cythonize(
        ["app/Script/*.py", "app/view/*.py","app/view/Ui/*.py"],
        compiler_directives={"language_level": "3"},
        build_dir="build"
    ),
    # package_data={"config": ["*.py"], "task": ["*.py"]},
)


path = "build/lib.win-amd64-cpython-311/"
name = ".cp311-win_amd64"
# os.remove("D:/Desktop/app/view/AppWindow.pyd")
# os.remove("D:/Desktop/ChronoSnowScript/dist/app-备份/app/view/AppWindow.pyd")
shutil.copy(f"{path}AppWindow{name}.pyd", "D:/Desktop/app/view/AppWindow.pyd")
shutil.move(f"{path}AppWindow{name}.pyd", "D:/Desktop/ChronoSnowScript/dist/app-备份/app/view/AppWindow.pyd")

shutil.copy(f"{path}ClientServices{name}.pyd", "D:/Desktop/app/view/ClientServices.pyd")
shutil.move(f"{path}ClientServices{name}.pyd", "D:/Desktop/ChronoSnowScript/dist/app-备份/app/view/ClientServices.pyd")

# os.remove("D:/Desktop/app/view/Public.pyd")
# os.remove("D:/Desktop/ChronoSnowScript/dist/app-备份/app/view/Public.pyd")
shutil.copy(f"{path}Public{name}.pyd", "D:/Desktop/app/view/Public.pyd")
shutil.move(f"{path}Public{name}.pyd", "D:/Desktop/ChronoSnowScript/dist/app-备份/app/view/Public.pyd")

# os.remove("D:/Desktop/app/Script/BasicFunctional.pyd")
# os.remove("D:/Desktop/ChronoSnowScript/dist/app-备份/app/Script/BasicFunctional.pyd")
shutil.copy(f"{path}BasicFunctional{name}.pyd", "D:/Desktop/app/Script/BasicFunctional.pyd")
shutil.move(f"{path}BasicFunctional{name}.pyd", "D:/Desktop/ChronoSnowScript/dist/app-备份/app/Script/BasicFunctional.pyd")

# os.remove("D:/Desktop/app/Script/Task.pyd")
# os.remove("D:/Desktop/ChronoSnowScript/dist/app-备份/app/Script/Task.pyd")
shutil.copy(f"{path}Task{name}.pyd", "D:/Desktop/app/Script/Task.pyd")
shutil.move(f"{path}Task{name}.pyd", "D:/Desktop/ChronoSnowScript/dist/app-备份/app/Script/Task.pyd")

# os.remove("D:/Desktop/app/view/Ui/HomeWindow.pyd")
# os.remove("D:/Desktop/ChronoSnowScript/dist/app-备份/app/view/Ui/HomeWindow.pyd")
shutil.copy(f"{path}HomeWindow{name}.pyd", "D:/Desktop/app/view/Ui/HomeWindow.pyd")
shutil.move(f"{path}HomeWindow{name}.pyd", "D:/Desktop/ChronoSnowScript/dist/app-备份/app/view/Ui/HomeWindow.pyd")

# os.remove("D:/Desktop/app/view/Ui/MainWindow.pyd")
# os.remove("D:/Desktop/ChronoSnowScript/dist/app-备份/app/view/Ui/MainWindow.pyd")
shutil.copy(f"{path}MainWindow{name}.pyd", "D:/Desktop/app/view/Ui/MainWindow.pyd")
shutil.move(f"{path}MainWindow{name}.pyd", "D:/Desktop/ChronoSnowScript/dist/app-备份/app/view/Ui/MainWindow.pyd")

# os.remove("D:/Desktop/app/view/Ui/RunWindow.pyd")
# os.remove("D:/Desktop/ChronoSnowScript/dist/app-备份/app/view/Ui/RunWindow.pyd")
shutil.copy(f"{path}RunWindow{name}.pyd", "D:/Desktop/app/view/Ui/RunWindow.pyd")
shutil.move(f"{path}RunWindow{name}.pyd", "D:/Desktop/ChronoSnowScript/dist/app-备份/app/view/Ui/RunWindow.pyd")

# os.remove("D:/Desktop/app/view/Ui/ScriptWindow.pyd")
# os.remove("D:/Desktop/ChronoSnowScript/dist/app-备份/app/view/Ui/ScriptWindow.pyd")
shutil.copy(f"{path}ScriptWindow{name}.pyd", "D:/Desktop/app/view/Ui/ScriptWindow.pyd")
shutil.move(f"{path}ScriptWindow{name}.pyd", "D:/Desktop/ChronoSnowScript/dist/app-备份/app/view/Ui/ScriptWindow.pyd")

shutil.copy(f"{path}LoginWindow{name}.pyd", "D:/Desktop/app/view/Ui/LoginWindow.pyd")
shutil.move(f"{path}LoginWindow{name}.pyd", "D:/Desktop/ChronoSnowScript/dist/app-备份/app/view/Ui/LoginWindow.pyd")


