project (HelloKitty CXX)
include_directories(../include
					${OPENCVINCLUDEDIR})

set(SOURCES_FILES	./main.cpp
					)
					
add_executable(${PROJECT_NAME} ${SOURCES_FILES})

target_link_libraries(${PROJECT_NAME} 
	debug 		${OPENCVLIBDIR}/opencv_world310d.lib
	optimized	${OPENCVLIBDIR}/opencv_world310.lib
	)