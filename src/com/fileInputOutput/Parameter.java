package com.fileInputOutput;

public class Parameter {

    private double C;
    private int maxIteration;
    private double eps;
    private double C1;

    public Parameter(double C, int maxIteration, double eps) {
        this.C = C;
        this.maxIteration = maxIteration;
        this.eps = eps;
    }

    public Parameter(double C, double C1, int maxIteration, double eps) {
        this.C = C;
        this.C1 = C1;
        this.maxIteration = maxIteration;
        this.eps = eps;
    }

    public double getC() {
        return C;
    }

    public void setC(double c) {
        C = c;
    }

    public int getMaxIteration() {
        return maxIteration;
    }

    public void setMaxIteration(int maxIteration) {
        this.maxIteration = maxIteration;
    }

    public double getEps() {
        return eps;
    }

    public void setEps(double eps) {
        this.eps = eps;
    }

    public double getC1() {
        return C1;
    }

    public void setC1(double c1) {
        C1 = c1;
    }

}
