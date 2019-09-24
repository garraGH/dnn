target("utils")
    set_kind("static")
    add_headerfiles("*.h")
    add_files("*.cpp")
    before_build(function(target)
        target:add(find_packages("cuda"))
    end)

