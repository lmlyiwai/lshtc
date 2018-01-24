package com.DeepNetworkWithSVM;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.sparseVector.DataPoint;

public class Linear {
    private static Random random = new Random();

    public static double[] train(DataPoint[][] prob, int[] y, Parameter param, int dim) {
        int l = prob.length;
        int w_size = dim;
        int[] index = new int[l];
        double[] alpha = new double[l];
        int active_size = l;
        int i, s, iter = 0;
        double C, d, G;
        double[] QD = new double[l];

        double PG;
        double PGmax_old = Double.POSITIVE_INFINITY;
        double PGmin_old = Double.NEGATIVE_INFINITY;
        double PGmax_new, PGmin_new;

        double[] w = new double[w_size];

        for (i = 0; i < l; i++) {
            alpha[i] = 0;
        }

        for (i = 0; i < l; i++) {
            QD[i] = 0;
            for (DataPoint dp : prob[i]) {
                double val = dp.value;
                QD[i] += val * val;
                w[dp.index - 1] += y[i] * alpha[i] * val;
            }
            index[i] = i;
        }

        while (iter < param.getMaxIteration()) {
            PGmax_new = Double.NEGATIVE_INFINITY;
            PGmin_new = Double.POSITIVE_INFINITY;

            for (i = 0; i < active_size; i++) {
                int j = i + random.nextInt(active_size - i);
                swap(index, i, j);
            }

            for (s = 0; s < active_size; s++) {
                i = index[s];
                G = 0;
                int yi = y[i];

                for (DataPoint xi : prob[i]) {
                    G += w[xi.index - 1] * xi.value;
                }

                G = G * yi - 1;
                C = param.getC();

                PG = 0;
                if (alpha[i] == 0) {
                    if (G > PGmax_old) {
                        active_size--;
                        swap(index, s, active_size);
                        s--;
                        continue;
                    } else if (G < 0) {
                        PG = G;
                    }
                } else if (alpha[i] == C) {
                    if (G < PGmin_old) {
                        active_size--;
                        swap(index, s, active_size);
                        s--;
                        continue;
                    } else if (G > 0) {
                        PG = G;
                    }
                } else {
                    PG = G;
                }

                PGmax_new = Math.max(PGmax_new, PG);
                PGmin_new = Math.min(PGmin_new, PG);

                if (Math.abs(PG) > 1.0e-12) {
                    double alpha_old = alpha[i];
                    alpha[i] = Math.min(Math.max((alpha[i] - (G / QD[i])), 0.0), C);
                    d = (alpha[i] - alpha_old) * yi;

                    for (DataPoint xi : prob[i]) {
                        w[xi.index - 1] += d * xi.value;
                    }
                }

            }

            iter++;
            if (PGmax_new - PGmin_new <= param.getEps()) {
                if (active_size == l) {
                    break;
                } else {
                    active_size = l;
                    PGmax_old = Double.POSITIVE_INFINITY;
                    PGmin_old = Double.NEGATIVE_INFINITY;
                    continue;
                }
            }
            PGmax_old = PGmax_new;
            PGmin_old = PGmin_new;
            if (PGmax_old <= 0) PGmax_old = Double.POSITIVE_INFINITY;
            if (PGmin_old >= 0) PGmin_old = Double.NEGATIVE_INFINITY;
        }
        return alpha;
    }

    public static double[] getW(DataPoint[][] x, int[] y, double[] alpha, int dim) {
        double[] w = new double[dim];
        for (int i = 0; i < x.length; i++) {
            if (alpha[i] > 0) {
                double scale = y[i] * alpha[i];
                for (DataPoint dp : x[i]) {
                    w[dp.index - 1] += scale * dp.value;
                }
            }
        }
        return w;
    }

    private static void swap(int[] index, int i, int j) {
        int temp = index[i];
        index[i] = index[j];
        index[j] = temp;
    }
}
