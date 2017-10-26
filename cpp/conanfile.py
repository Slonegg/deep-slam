from conans import ConanFile


class DeepSlamConan(ConanFile):
    generators = "cmake", "cmake_multi"
    default_options = \
        "Boost:shared=True", \
        "opencv:with_contrib=True", \
        "opencv:with_vtk=False", \
        "opencv:with_openni2=True"
    requires = \
        "gtest/1.7.0@lasote/stable", \
        "spdlog/0.13.0@memsharded/stable"
    settings = "os", "compiler", "build_type", "arch"

    def imports(self):
        self.copy("*.dll", dst="bin", src="bin")
        self.copy("*.ini", dst="bin", src="bin")
        self.copy("*.so", dst="lib", src="bin")
        self.copy("*.so.*", dst="lib", src="bin")
