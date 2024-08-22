.PHONY: all

all: \
	radial_cart_pole_svg \
	linear_cart_pole_svg \
	radial_cart_pole_helper_svg \

build_dir:
	mkdir -p build

linear_cart_pole_svg: build_dir
	latex -output-directory build tex/linear_cart_pole.tex
	dvisvgm --no-fonts build/linear_cart_pole.dvi -o svg/linear_cart_pole.svg

radial_cart_pole_svg: build_dir
	latex -output-directory build tex/radial_cart_pole.tex
	dvisvgm --no-fonts build/radial_cart_pole.dvi -o svg/radial_cart_pole.svg

radial_cart_pole_helper_svg: build_dir
	latex -output-directory build tex/radial_cart_pole_helper.tex
	dvisvgm --no-fonts build/radial_cart_pole_helper.dvi -o svg/radial_cart_pole_helper.svg


clean:
	rm -rf build
	cd tex && latexmk -CA
