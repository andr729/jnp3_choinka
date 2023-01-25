cmake_minimum_required (VERSION 3.8)

project ("Direct3D")

find_library(DIRECT3D d3d12)
if (NOT DIRECT3D)
    message(FATAL_ERROR "Could not find Direct3D.")
endif ()
find_library(DXGI dxgi)
if (NOT DXGI)
    message(FATAL_ERROR "Could not find DXGI.")
endif ()

set (CMAKE_WIN32_EXECUTABLE "True")
set (CMAKE_CXX_FLAGS 
"/Wall /std:c++20 /DUNICODE /TP /Zc:__cplusplus /EHs /MT")



# Szukanie kompilatora plików .hlsl
find_program (FXC fxc.exe)
if (NOT FXC)
    message(FATAL_ERROR "Could not find fxc.exe")
endif ()

add_custom_target(
 HLSL_Shaders ALL 
 COMMAND ${FXC} /T vs_5_1 /Vn vs_main /Fh vertex_shader.h VertexShader.hlsl
 COMMAND ${FXC} /T ps_5_1 /Vn ps_main /Fh pixel_shader.h PixelShader.hlsl
 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
 VERBATIM
)


set (SOURCE_FILES
    src/D3DApp.h src/D3DApp.cpp src/WinMain.h src/WinMain.cpp)

add_executable (Window ${SOURCE_FILES})

target_link_libraries(Window ${DIRECT3D})
target_link_libraries(Window ${DXGI})

add_dependencies(HLSL_Shaders Window)


