add_includedirs("pkg/include")
add_packagedirs("pkg/build")
add_subdirs("pkg_src/utils/xmake_cpu.lua", "pkg_src/elsa", "sandbox/xmake_cpu.lua")
