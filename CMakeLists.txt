cmake_minimum_required(VERSION 2.8.9)

project(textureBasedSegmentation)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "http://slicer.org/slicerWiki/index.php/Documentation/Nightly/Extensions/textureBasedSegmentation")
set(EXTENSION_CATEGORY "Examples")
set(EXTENSION_CONTRIBUTORS "Brandon Chan (Queen's University), Nuwan Perera (Queen's University), Mareena Mallory (Queen's University)")
set(EXTENSION_DESCRIPTION "Slicer module to enable segmentation of images based using machine learning and textural values")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/textureBasedSegmentation.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/textureBasedSegmentation/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a space separated string, a list or 'NA' if any

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(textureBasedSegmentation)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_CPACK})
