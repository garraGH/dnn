includes("../utils/xmake.lua")
includes("../../deps/modules/glfw/xmake.lua")
includes("../../deps/modules/imgui/xmake.lua")
includes("../../deps/modules/assimp/xmake.lua")
includes("../../deps/modules/osdialog/xmake.lua")
target("elsa")
    set_kind("static")
    add_includedirs("./", {public = true} )
    add_includedirs("../../deps/modules/stb", {public=true})
    add_files("./**.cpp")
    add_deps("utils")
    add_deps("glad")
    add_deps("glfw")
    add_deps("imgui")
    add_deps("assimp")
    add_deps("osdialog")
    add_cxflags("$(shell pkg-config --cflags cuda)")
--    before_build(function(target)
--        target:add(find_packages("cuda"))
--    end)

