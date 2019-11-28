includes("../src/elsa/xmake.lua")
add_syslinks("dl", "pthread", "X11", "OpenGL")
target("sandbox")
    set_kind("binary")
    add_files("./**.cpp")
    add_deps("elsa")
    if is_plat("linux") then
        add_ldflags("$(shell pkg-config --libs gtk+-3.0)")
    end
