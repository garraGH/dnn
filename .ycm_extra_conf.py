import os
import ycm_core


def Subdirectories(directory):
  res = []
  for path, subdirs, files in os.walk(directory):
    for name in subdirs:
      item = os.path.join(path, name)
      res.append(item)
  return res

def IncludeFlagsOfSubdirectory( flags, working_directory ):
  if not working_directory:
    return list( flags )
  new_flags = []
  make_next_include_subdir = False
  path_flags = [ '-ISUB']
  for flag in flags:
    # include the directory of flag as well
    new_flag = [flag.replace('-ISUB', '-I')]

    if make_next_include_subdir:
      make_next_include_subdir = False
      for subdir in Subdirectories(os.path.join(working_directory, flag)):
        new_flag.append('-I')
        new_flag.append(subdir)

    for path_flag in path_flags:
      if flag == path_flag:
        make_next_include_subdir = True
        break

      if flag.startswith( path_flag ):
        path = flag[ len( path_flag ): ]
        for subdir in Subdirectories(os.path.join(working_directory, path)):
            basename = os.path.basename(subdir)
            if basename=="include" or basename=="inc":
                new_flag.append('-I' + subdir)
        break

    new_flags =new_flags + new_flag
  return new_flags

flags = [
        '-Wall', 
        '-Wextra', 
        '-Werror', 
        '-Wno-long-long', 
        '-Wno-variadic-macros', 
        '-fexceptions', 
        '-ferror-limit=1000', 
        '-DNDEBUG', 
        '-xc++', 
        '-std=c++11', 
        '-isystem/usr/include/', 
        '-isystem/usr/local/cuda/include/',
        '-ISUBdeps/',
        '-I./src/elsa',
        '-I./src/utils/',
        '-I./deps/modules/glm/',
        '-I./deps/modules/spdlog/include/',
        '-I./deps/modules/imgui/',
        '-I./deps/modules/imgui/examples/',
        '-I./deps/modules/glfw/deps/',
        '-I./deps/modules/stb/',
        '-I./deps/modules/assimp/include/',
        '-I./deps/modules/osdialog/',
        ]

flags = IncludeFlagsOfSubdirectory(flags, "./" )
print(flags)

SOURCE_EXTENSIONS = ['.cpp', 'cxx', 'cc', '.c', '.h', '.hpp', 'cu']

def FlagsForFile(filename, **kwargs):
    return {
            'flags': flags, 
            'do_cache': True
            }
