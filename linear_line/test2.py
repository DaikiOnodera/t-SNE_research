#!/usr/bin/env python
# encoding:utf-8

import matplotlib.pyplot as plt

def main():
    (x, y) = (0, 0)
    plt.ion()

    while(y!=10):
        line, = plt.plot(x, y, "ro", label="y=x")
        line.set_ydata(y)
        plt.title("Graph")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid()
        plt.xlim([0,10])
        plt.ylim([0,10])
        plt.draw()
        plt.clf()
        x += 1
        y = x

    plt.close()

if __name__ == "__main__":
    main()
