#!/usr/bin/python

import sys, getopt

def main(argv):
    print(argv)
    xshift = float(argv[3])
    yshift = float(argv[4])
    x_file = argv[0]
    y_file = argv[1]
    z_file = argv[2]
    o_file = "output_mode.dat"
    with open(x_file) as xh:
        with open(y_file) as yh:
            with open(z_file) as zh:
                with open(o_file, "w") as of:
                    xlines = xh.readlines()
                    ylines = yh.readlines()
                    zlines = zh.readlines()
                    for i in range(len(xlines)):
                        xsplit = xlines[i].split("  ")
                        ysplit = ylines[i].split("  ")
                        zsplit = zlines[i].split("  ")
                        x = str(float(xsplit[1]) - xshift)
                        y = str(float(xsplit[0]) - yshift)
                        xsplit[2] = xsplit[2].replace("\n", "")
                        ysplit[2] = ysplit[2].replace("\n", "")
                        zsplit[2] = zsplit[2].replace("\n", "")
                        print( x + " " + y + " " + xsplit[2] + " " + ysplit[2] + " " + zsplit[2])
                        of.write( x + " " + y + " " + xsplit[2] + " " + ysplit[2] + " " + zsplit[2] + "\n")


if __name__ == "__main__":
   main(sys.argv[1:])