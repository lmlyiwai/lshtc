package com.tools;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;

import com.sparseVector.DataPoint;

public class Matrix {
	/**
	 * 打印矩阵
	 */
	public static void showMatrix(double[][] matrix) {
		if(null == matrix) {
			return;
		}
		for(int i = 0; i < matrix.length; i++) {
			for(int j = 0; j < matrix[0].length; j++) {
				System.out.print(matrix[i][j] + " ");
			}
			System.out.println();
		}
	}
	
	/**
	 * 
	 */
	public static double[][] trans(double[][] matrix) {
		int row = matrix.length;
		int col = matrix[0].length;
		
		double[][] tmatrix = new double[col][row];
		for(int i = 0; i < matrix.length; i++) {
			for(int j = 0; j < matrix[i].length; j++) {
				tmatrix[j][i] = matrix[i][j];
			}
		}
		return tmatrix;
	}
	
	/**
	 * 输入矩阵a,b，a,b中有null是返回null
	 */
	public static double[][] multi(double[][] a, double[][] b) {
		if(a == null || b == null) {
			return null;
		}
		int rowa = a.length;
		int cola = a[0].length;
		int rowb = b.length;
		int colb = b[0].length;
		if(cola != rowb) {
			return null;
		}
		
		double[][] mul = new double[rowa][colb];
		for(int i = 0; i < mul.length; i++) {
			for(int j = 0; j < mul[0].length; j++) {
				for(int k = 0; k < cola; k++) {
					mul[i][j] += a[i][k] * b[k][j];
				}
			}
		}
		return mul;
	}
	
	/**
	 * 输入矩阵a,a为null时返回Double.NaN
	 */
	public static double det(double[][] a) {
		if(null == a) {
			return Double.NaN;
		}
		int row = a.length;
		int col = a[0].length;
		
		
		if(row != col) {
			return Double.NaN;
		}
		
		if(row == 1) {
			return a[0][0];
		} else {
			double sum = 0;
			for(int i = 0; i < a.length; i++) {
				double f = Math.pow(-1, i);
				sum = sum + f * a[0][i] * det(exclued(a, 0, i));
			}
			return sum;
		}
	}
	
	/**
	 * 
	 */
	public static double[][] identityMatrix(int n) {
		double[][] matrix = new double[n][n];
		for(int i = 0; i < n; i++) {
			matrix[i][i] = 1;
		}
		return matrix;
	}
	
	/**
	 * 原地缩放矩阵，矩阵本身值发生改变
	 */
	public static void scaleMatrix(double[][] mat, double scale) {
		for(int i = 0; i < mat.length; i++) {
			for(int j = 0; j < mat[i].length; j++) {
				mat[i][j] = scale * mat[i][j];
			}
		}
	}
	
	/**
	 * 去除矩阵i行j列元素
	 */
	public static double[][] exclued(double[][] a, int i, int j) {
		if(null == a || i >= a.length || j >= a[0].length) {
			return null;
		}
		
		int row = a.length;
		int col = a[0].length;
		
		double[] all = new double[(row - 1) * (col - 1)];
		int counter = 0;
		for(int m = 0; m < row; m++) {
			for(int n = 0; n < col; n++) {
				if(m != i && n != j) {
					all[counter] = a[m][n];
					counter = counter + 1;
				}
			}
		}
		
		double[][] result = new double[row - 1][col - 1];
		counter = 0;
		for(int m = 0; m < result.length; m++) {
			for(int n = 0; n < result.length; n++) {
				result[m][n] = all[counter];
				counter = counter + 1;
			}
		}
		return result;
	}
	
	/**
	 * 代数余子式
	 */
	public static double[][] algebraciComplement(double[][] a) {
		if(null == a) {
			return null;
		}
		
		int row = a.length;
		int col = a[0].length;
		double[][] result = new double[row][col];
		
		for(int i = 0; i < row; i++) {
			for(int j = 0; j < col; j++) {
				double sign = Math.pow(-1, i+j);
				double[][] comple = exclued(a, i, j);
				double detcom = det(comple);
				result[i][j] = sign * detcom;
			}
		}
		
		double[][] r = trans(result);
		return r;
	}
	
	/**
	 * 求矩阵逆矩阵，无解时返回null
	 */
	public static double[][] inv(double[][] a) {
		if(null == a) {
			return null;
		}
		
		double deta = det(a);
		if(deta == 0) {
			return null;
		}
		
		double[][] acom = algebraciComplement(a);
		
		scaleMatrix(acom, 1 / deta);
		
		return acom;
	}
	
	/**
	 * 
	 */
	public static double[][] matrixSub(double[][] a, double[][] b) {
		int rowa = a.length;
		int cola = a[0].length;
		
		int rowb = b.length;
		int colb = b[0].length;
		
		if(rowa != rowb || cola != colb) {
			return null;
		}
		
		double[][] result = new double[rowa][colb];
		for(int i = 0; i < rowa; i++) {
			for(int j = 0; j < cola; j++) {
				result[i][j] = a[i][j] - b[i][j];
			}
		}
		return result;
	}
	
	/**
	 * 
	 */
	public static double l2norm(double[][] a) {
		double sum = 0;
		for(int i = 0; i < a.length; i++) {
			for(int j = 0; j < a[i].length; j++) {
				sum = sum + a[i][j] * a[i][j];
			}
		}
		double result = Math.pow(sum, 0.5);
		return result;
	}
	
	/**
	 * 
	 */
	public static double[][] matrixAdd(double[][] a, double[][] b) {
		int rowa = a.length;
		int cola = a[0].length;
		
		int rowb = b.length;
		int colb = b[0].length;
		
		if(rowa != rowb || cola != colb) {
			return null;
		}
		
		double[][] result = new double[rowa][colb];
		for(int i = 0; i < rowa; i++) {
			for(int j = 0; j < cola; j++) {
				result[i][j] = a[i][j] + b[i][j];
			}
		}
		return result;
	}
	
	/**
	 * 
	 */
	public static double[] getMatrixColumn(double[][] matrix, int col) {
		double[] result = new double[matrix.length];
		for(int i = 0; i < matrix.length; i++) {
			result[i] = matrix[i][col];
		}
		return result;
	}
	
	/**
	 * @throws IOException 
	 * 
	 */
	public static void writeMatrixToFile(double[][] mat, String filename) throws IOException {
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(filename)));
		for(int i = 0; i < mat.length; i++) {
			String line = new String();
			for(int j = 0; j < mat[i].length - 1; j++) {
				line += mat[i][j] + " ";
			}
			line += mat[i][mat[i].length - 1] + "\n";
			out.write(line);
		}
		out.close();
	}
	
	/**
	 * @throws IOException 
	 * 
	 */
	public static double[][] readMatrixFromFile(String filename) throws IOException {
		List<double[]> list = new ArrayList<double[]>();
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(filename)));
		String line = null;
		String[] splits = null;
		double[] temp = null;
		while((line = in.readLine()) != null) {
			splits = line.split("\\s+|\r|\n|\t");
			temp = new double[splits.length];
			for(int j = 0; j < temp.length; j++) {
				temp[j] = Double.parseDouble(splits[j]);
			}
			list.add(temp);
		}
		in.close();
		
		double[][] result = new double[list.size()][];
		for(int i = 0; i < result.length; i++) {
			result[i] = list.get(i);
		}
		return result;
	}
	
	/**
	 * 
	 */
	public static double findVecMax(double[] vec) {
		double max = Double.NEGATIVE_INFINITY;
		for(int i = 0; i < vec.length; i++) {
			if(vec[i] > max) {
				max = vec[i];
			}
		}
		return max;
	}
	
	/**
	 * 
	 */
	public static int findLabel(int[] labels, int label) {
		int index = -1;
		for(int i = 0; i < labels.length; i++) {
			if(labels[i] == label) {
				index = i;
				break;
			}
		}
		return index;
	}
	
	/**
	 * 
	 */
	public static double[] vecAdd(double[] a, double[] b) {
		if(a.length != b.length) {
			return null;
		}
		
		double[] sum = new double[a.length];
		for(int i = 0; i < sum.length; i++) {
			sum[i] = a[i] + b[i];
		}
		return sum;
	}
	
	/**
	 * 矩阵与稀疏向量相乘，返回值为行向量
	 */
	public static double[] multi(double[][] w, DataPoint[] x) {
		double[] result = new double[w.length];
		for(int i = 0; i < w.length; i++) {
			double sum = 0.0;
			for(int j = 0; j < x.length; j++) {
				sum += w[i][x[j].index-1] * x[j].value;
			}
			result[i] = sum;
		}
		return result;
	}
	
	/**
	 * 
	 */
	public static double multi(double[] fullVec, DataPoint[] sparseVec) {
		double sum = 0.0;
		for(int i = 0; i < sparseVec.length; i++) {
			sum += fullVec[sparseVec[i].index - 1] * sparseVec[i].value;
		}
		return sum;
	}
	
	/**
	 * 矩阵与列向量相乘
	 */
	public static double[] multi(double[][] mat, double[] vec) {
		if(null == mat || null == vec) {
			return null;
		}
		int r = mat.length;
		int c = mat[0].length;
		
		int vecr = vec.length;
		
		if(c != vecr) {
			return null;
		}
		
		double[] result = new double[r];
		for(int i = 0; i < r; i++) {
			double sum = 0;
			for(int j = 0; j < c; j++) {
				sum += mat[i][j] * vec[j];
			}
			result[i] = sum;
		}
		return result;
	}
}
