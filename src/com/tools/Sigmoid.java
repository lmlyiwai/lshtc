package com.tools;

import com.sparseVector.SparseVector;

public class Sigmoid {
	/**
	 *	f(x) = 1 / (1 + e ^ (-a * x)); 
	 */
	public static double sigmoid(double x, double a) {
		double it1 = -a * x;
		double it2 = 1 + Math.exp(it1);
		return 1 / it2;
	}
	
	/**
	 * f(x) = (exp(ax) - exp(-ax)) / (exp(ax) + exp(-ax)) 
	 */
	public static double tanhx(double x, double a) {
		double p = a * x;
		double n = -a * x;
		double it1 = Math.exp(p) - Math.exp(n);
		double it2 = Math.exp(p) + Math.exp(n);
		return it1 / it2;
	}
	
	/**
	 * 对m中的每个值进行变换
	 */
	public static void sigmoid(double[][] m, double a) {
		if(m == null) {
			return;
		}
		
		for(int i = 0; i < m.length; i++) {
			for(int j = 0; j < m[i].length; j++) {
				m[i][j] = sigmoid(m[i][j], a);
			}
		}
	}
	
	/**
	 * 对 m进行tanhx变换 
	 */
	public static void tanhx(double[][] m, double a) {
		if(m == null) {
			return;
		}
		
		for(int i = 0; i < m.length; i++) {
			for(int j = 0; j < m[i].length; j++) {
				m[i][j] = tanhx(m[i][j], a);
			}
		}
	}
	
	/**
	 * 
	 */
	public static double entropy(double[] vec) {
		double sum = 0;
		double[] svec = sigmoid(vec);
		for(int i = 0; i < vec.length; i++) {
			sum += svec[i] * Math.log(svec[i]);
		}
		return -sum;
	}
	
	/**
	 * 
	 */
	public static double[] sigmoid(double[] vec) {
		double[] s = new double[vec.length];
		for(int i = 0; i < s.length; i++) {
			s[i] = 1.0 / (1.0 + Math.exp(-vec[i]));
		}
		return s;
	}
	
	/**
	 * 
	 */
	public static double[] entropy(double[][] mat) {
		double[] ent = new double[mat.length];
		for(int i = 0; i < ent.length; i++) {
			ent[i] = entropy(mat[i]);
		}
		return ent;
	}
	
	/**
	 * 
	 */
	public static void scale(double[][] pv) {
		for(int i = 0; i < pv.length; i++) {
			double inp = SparseVector.innerProduct(pv[i], pv[i]);
			inp = Math.pow(inp, 0.5);
			for(int j = 0; j < pv[i].length; j++) {
				pv[i][j] /= inp;
			}
		}
	}
}
