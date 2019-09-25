
target("elsa")
    set_kind("static")
    add_headerfiles("*.h","**/*.h")
    add_files("**/*.cpp")
    add_packages("utils")
