
target("elsa")
    set_kind("static")
    add_headerfiles("*.h", "app/*.h", "event/*.h", "window/*.h")
    add_files("**/*.cpp")
    add_packages("utils", "glfw3")
    before_build(function(target)
        target:add(find_packages("cuda"))
    end)

