package com.tools;

public class Kernel {
	/**
	 * 
	 */
	public static double[] polynomial(double[] d) {
		if(d == null) {
			return null;
		}
		
		int n = d.length;
		n = n + n * (n - 1) / 2;
		double[] result = new double[n];
		int counter = 0;
		for(int i = 0; i < d.length; i++) {
			result[counter++] = d[i] * d[i];
		}
		
		for(int i = 0; i < d.length; i++) {
			for(int j = i + 1; j < d.length; j++) {
				result[counter++] = 2 * d[i] * d[j];
			}
		}
		
		double sum = 0;
		for(int i = 0; i < result.length; i++) {
			sum += result[i] * result[i];
		}
		
		double norm = Math.pow(sum, 0.5);
		for(int i = 0; i < result.length; i++) {
			result[i] = result[i] / norm;
		}
		return result;
	}
	
	/**
	 * 
	 */
	public static double[][] map(double[][] old) {
		double[][] newMatrix = new double[old.length][];
		for(int i = 0; i < newMatrix.length; i++) {
			newMatrix[i] = polynomial(old[i]);
		}
		return newMatrix;
	}
}
