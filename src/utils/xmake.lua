target("utils")
    set_kind("static")
    add_includedirs(".", {public=true})
    add_files("*.cpp")
    add_cxflags("$(shell pkg-config --cflags cuda)")
--    before_build(function(target)
--        target:add(find_packages("cuda"))
--    end)

