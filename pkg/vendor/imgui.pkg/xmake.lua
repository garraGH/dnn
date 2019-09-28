option("use_glad")
    set_default(true)
    set_showmenu(true)
    add_defines("USE_GLAD")

option("use_gl3w")
    set_default(false)
    set_showmenu(true)
    add_defines("USE_GL3W")

option("imgui")
    set_showmenu(true)
    set_category("package")
    add_links("imgui_glad", "glad")
--    on_load(function(target)
--        if(has_config("use_glad")) then
--            target:add_links("glad", "imgui_glad")
--        elseif(has_config("use_gl3w")) then
--            target:add_links("gl3w", "imgui_gl3w")
--        end
--    end)
    add_linkdirs("lib")
    add_includedirs("include")
