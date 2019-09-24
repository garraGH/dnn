
add_packagedirs("../pkg/build/")
target("sandboxapp")
    set_kind("binary")
    add_files("*.cpp")
    add_packages("elsa")
