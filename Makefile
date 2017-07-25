CXX = g++
CXXFLAGS = $(OPT) $(DEBUG) $$(pkgconf --cflags eigen3) $$(pkgconf --cflags gtkmm-3.0)
OPT = -O
DEBUG = -g
LIBS = $$(pkgconf --libs eigen3) $$(pkgconf --libs gtkmm-3.0)

all: nn-bp-gtkmm

nn-bp-gtkmm: nn-bp-gtkmm.cc nn-bp.h nn-bp.tcc nn-bp-data.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIBS)
