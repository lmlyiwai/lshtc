package com.deep;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;

public class Matrix {
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
	 * 
	 */
	public static double[][] multi(double[][] a, double[][] b) {
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
	 * 
	 */
	public static double det(double[][] a) {
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
	 * 
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
	 * 逆矩阵
	 */
	public static double[][] inv(double[][] a) {
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
	 * 原地相加，和保存在a中
	 */
	public static void matrixLocalAdd(double[][] a, double[][] b) {
		int rowa = a.length;
		int cola = a[0].length;
		
		int rowb = b.length;
		int colb = b[0].length;
		
		if(rowa != rowb || cola != colb) {
			return;
		}
		
		for(int i = 0; i < rowa; i++) {
			for(int j = 0; j < cola; j++) {
				a[i][j] = a[i][j] + b[i][j];
			}
		}
	}
	
	/**
	 * 和保留在a中，b的一行中某个元素为0，则改行全为0
	 */
	public static void sparseMatAdd(double[][] a, double[][] b) {
		int rowa = a.length;
		int cola = a[0].length;
		
		int rowb = b.length;
		int colb = b[0].length;
		
		if(rowa != rowb || cola != colb) {
			return;
		}
		
		for(int i = 0; i < rowa; i++) {
			if(b[i][0] == 0) {
				continue;
			}
			for(int j = 0; j < cola; j++) {
				a[i][j] = a[i][j] + b[i][j];
			}
		}
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
	 * 
	 */
	public static void printMatrix(double[][] matrix) {
		if(matrix == null) {
			return;
		}
		
		for(int i = 0; i < matrix.length; i++) {
			for(int j = 0; j < matrix[i].length; j++) {
				System.out.print(matrix[i][j] + " ");
			}
			System.out.println();
		}
		System.out.println();
	}
	
	/**
	 *  行向量与矩阵相乘
	 */
	public static double[] multi(double[] vector, double[][] matrix) {
		if(vector == null || matrix == null) {
			return null;
		}
		
		int lvector = vector.length;
		int rmatrix = matrix.length;
		if(lvector != rmatrix) {
			return null;
		}
		
		int c = matrix[0].length;
		double[] result = new double[c];
		for(int i = 0; i < matrix[0].length; i++) {
			result[i] = 0;
			for(int j = 0; j < vector.length; j++) {
				result[i] += vector[j] * matrix[j][i];
			}
		}
		return result;
	}
	
	/**
	 * 行向量相减
	 */
	public static double[] vectorSub(double[] a, double[] b) {
		if(a == null || b == null || a.length != b.length) {
			return null;
		}
		
		double[] result = new double[a.length];
		for(int i = 0; i < a.length; i++) {
			result[i] = a[i] - b[i];
		}
		return result;
	}
	
	/**
	 * 行向量与列向量相乘,a为行向量,b为列向量
	 */
	public static double[][] multi(double[] a, double[] b) {
		if(a == null || b == null) {
			return null;
		}
		
		int row = a.length;
		int col = b.length;
		double[][] result = new double[row][col];
		for(int i = 0; i < row; i++) {
			for(int j = 0; j < col; j++) {
				result[i][j] = a[i] * b[j];
			}
		}
		return result;
	} 
	
	/**
	 * 向量a中包含众多0
	 */
	public static double[][] sparseMulti(double[] a, double[] b) {
		if(a == null || b == null) {
			return null;
		}
		
		int row = a.length;
		int col = b.length;
		double[][] result = new double[row][col];
		for(int i = 0; i < row; i++) {
			if(a[i] == 0) {
				continue;
			}
			for(int j = 0; j < col; j++) {
				result[i][j] = a[i] * b[j];
			}
		}
		return result;		
	}
	
	/**
	 * 矩阵中每个元素与系数相乘 ，返回新矩阵，原矩阵不变
	 */
	public static double[][] scale(double[][] matrix, double s) {
		if(matrix == null) {
			return null;
		}
		
		double[][] r = new double[matrix.length][matrix[0].length];
		for(int i = 0; i < r.length; i++) {
			for(int j = 0; j < r[0].length; j++) {
				r[i][j] = matrix[i][j] * s;
			}
		}
		return r;
	}
	
	/**
	 * 原矩阵本身改变
	 */
	public static void localScale(double[][] matrix, double s) {
		for(int i = 0; i < matrix.length; i++) {
			for(int j = 0; j < matrix[i].length; j++) {
				matrix[i][j] = matrix[i][j] * s;
			}
		}
	}
	
	/**
	 * matrix一行中存在0，则整行为0.
	 */
	public static void sparseLocalScale(double[][] matrix, double s) {
		for(int i = 0; i < matrix.length; i++) {
			if(matrix[i][0] == 0) {
				continue;
			}
			for(int j = 0; j < matrix[i].length; j++) {
				matrix[i][j] = matrix[i][j] * s;
			}
		}
	}
	
	/**
	 * 随机初始化矩阵
	 */
	public static void randInitMat(double[][] mat, double lb, double tb) {
		double length = Math.abs(lb - tb);
		for(int i = 0; i < mat.length; i++) {
			for(int j = 0; j < mat[i].length; j++) {
				mat[i][j] = Math.random() * length + lb;
			}
		}
	}
	
	/**
	 * 
	 */
	public static double innerProcuct(double[] a, double[] b) {
		if(a == null || b == null || a.length != b.length) {
			return Double.NaN;
		}
		
		double sum = 0;
		for(int i = 0; i < a.length; i++) {
			sum += a[i] * b[i];
		}
		return sum;
	}
	
	/**
	 * @throws IOException 
	 * 
	 */
	public static double[][] readArray(String filename) throws IOException {
		BufferedReader in = new BufferedReader(new InputStreamReader(
				new FileInputStream(filename)));
		String line = null;
		String[] splits = null;
		
		List<double[]> list = new ArrayList<double[]>();
		while((line = in.readLine()) != null) {
			splits = line.split("\\s+|\r|\n|\t");
			double[] d = new double[splits.length];
			for(int i = 0; i < d.length; i++) {
				d[i] = Double.parseDouble(splits[i]);
			}
			list.add(d);
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
	public static double[] multi(DataPoint[] x, double[][] w) {
		double[] result = new double[w[0].length];
		for(int i = 0; i < w[0].length; i++) {
			result[i] = 0;
			for(DataPoint dp : x) {
				result[i] += w[dp.index - 1][i] * dp.value;
			}
		}
		return result;
	}
	
	/**
	 * 
	 */
	public static double[][] fullMatrix(int[][] sparseMat, int dim) {
		int row = sparseMat.length;
		int col = dim;
		
		double[][] result = new double[row][col];
		for(int i = 0; i < row; i++) {
			for(int j = 0; j < col; j++) {
				result[i][j] = 0.1;
			}
			
			for(int j = 0; j < sparseMat[i].length; j++) {
				result[i][sparseMat[i][j]] = 0.9;
			}
		}
		return result;
	}
	
	/**
	 * 
	 */
	public static int uniqueLabels(int[][] labels) {
		Set<Integer> set = new HashSet<Integer>();
		for(int i = 0; i < labels.length; i++) {
			for(int j = 0; j < labels[i].length; j++) {
				set.add(labels[i][j]);
			}
		}
		int max = Integer.MIN_VALUE;
		Iterator<Integer> it = set.iterator();
		while(it.hasNext()) {
			int n = it.next();
			if(n > max) {
				max = n; 
			}
		}
		return (max+1);
	}
	
	/**
	 * f(x) = 1 / (1 + exp(-x)) 求逆
	 */
	public static double reverse(double y) {
		double x = -Math.log(1 / y - 1);
		return x;
	}
	
	/**
	 * 
	 */
	public static double[][] reverseMat(double[][] target) {
		double[][] result = new double[target.length][target[0].length];
		for(int i = 0; i < result.length; i++) {
			for(int j = 0; j < result[0].length; j++) {
				result[i][j] = reverse(target[i][j]);
			}
		}
		return result;
	}
	
	/**
	 * f(x) = 1 / (1 + exp(-x)); 
	 */
	public static double sigmoid(double x) {
		double y = 1 / (1 + Math.exp(-x));
		return y;
	}
	
	/**
	 * 
	 */
	public static double[][] sigmoidMat(double[][] mat) {
		double[][] result = new double[mat.length][mat[0].length];
		for(int i = 0; i < result.length; i++) {
			for(int j = 0; j < result[i].length; j++) {
				result[i][j] = sigmoid(mat[i][j]);
			}
		}
		return result;
	}
	
	/**
	 * 
	 */
	public static double[] sigmoidVec(double[] vector) {
		double[] result = new double[vector.length];
		for(int i = 0; i < result.length; i++) {
			result[i] = sigmoid(vector[i]);
		}
		return result;
	}
	/**
	 * 为矩阵扩展一列
	 */
	public static double[][] extendMat(double[][] mat, double bias) {
		double[][] result = new double[mat.length][mat[0].length+1];
		for(int i = 0; i < mat.length; i++) {
			int j;
			for(j = 0; j < mat[i].length; j++) {
				result[i][j] = mat[i][j];
			}
			result[i][j] = bias;
		}
		return result;
	}
	
	/**
	 * 
	 */
	public static double[] extendVec(double[] vec, double bias) {
		double[] result = new double[vec.length + 1];
		for(int i = 0; i < vec.length; i++) {
			result[i] = vec[i];
		}
		result[result.length - 1] = bias;
		return result;
	}
	
	/**
	 * 
	 */
	public static double[][] multi(DataPoint[][] xs, double[][] w) {
		double[][] result = new double[xs.length][];
		for(int i = 0; i < xs.length; i++) {
			result[i] = multi(xs[i], w);
		}
		return result;
	}
	
	/**
	 * 
	 */
	public static double[] sub(double d, double[] vec) {
		double[] result = new double[vec.length];
		for(int i = 0; i < result.length; i++) {
			result[i] = d - vec[i];
		}
		return result;
	}
	
	/**
	 * 
	 */
	public static double[] outProduct(double[] a, double[] b) {
		if(a.length != b.length) {
			return null;
		}
		double[] result = new double[a.length];
		for(int i = 0; i < result.length; i++) {
			result[i] = a[i] * b[i];
		}
		return result;
	}
	
	/**
	 * 
	 */
	public static double[] scaleVec(double[] vec, double scale) {
		double[] result = new double[vec.length];
		for(int i = 0; i < result.length; i++) {
			result[i] = vec[i] * scale;
		}
		return result;
	}
	
	/**
	 * 
	 */
	public static double[] cutVec(double[] vec) {
		double[] result = new double[vec.length - 1];
		for(int i = 0; i < vec.length - 1; i++) {
			result[i] = vec[i];
		}
		return result;
	}
	
	/**
	 * 
	 */
	public static void copyMat(double[][] from, double[][] to) {
		int fr = from.length;
		int fc = from[0].length;
		
		int tr = to.length;
		int tc = to[0].length;
		
		if(fr != tr || fc != tc) {
			return;
		}
		
		for(int i = 0; i < from.length; i++) {
			for(int j = 0; j < from[i].length; j++) {
				to[i][j] = from[i][j];
			}
		}
	}
	
	/**
	 * 单位矩阵
	 */
	public static void identifyMat(double[][] mat) {
		for(int i = 0; i < mat.length; i++) {
			for(int j = 0; j < mat[0].length; j++) {
				if(i == j) {
					mat[i][j] = 1;
				} else {
					mat[i][j] = 0;
				}
			}
		}
	}
	
	/**
	 * 
	 */
	public static double[][] cutMat(double[][] mat) {
		int row = mat.length;
		int col = mat[0].length;
		
		double[][] nmat = new double[row][col-1];
		for(int i = 0; i < row; i++) {
			for(int j = 0; j < col-1; j++) {
				nmat[i][j] = mat[i][j];
			}
		}
		return nmat;
	}
	
	/**
	 * 
	 */
	public static double maxMat(double[][] mat) {
		double max = Double.NEGATIVE_INFINITY;
		for(int i = 0; i < mat.length; i++) {
			for(int j = 0; j < mat[i].length; j++) {
				if(Math.abs(mat[i][j]) > max) {
					max = Math.abs(mat[i][j]);
				}
			}
		}
		return max;
	}
	
	/**
	 * 返回矩阵每一列最大值
	 */
	public static double[] colMax(double[][] mat) {
		double[] max = new double[mat[0].length];
		for(int i = 0; i < max.length; i++) {
			max[i] = Double.NEGATIVE_INFINITY;
		}
		
		for(int i = 0; i < mat[0].length; i++) {
			for(int j = 0; j < mat.length; j++) {
				if(Math.abs(mat[j][i]) > max[i]) {
					max[i] = Math.abs(mat[j][i]);
				}
			}
		}
		return max;
	}
	
	/**
	 * 对矩阵每列乘以相同系数
	 */
	public static double[][] scale(double[][] mat, double[] s) {
		double[][] result = new double[mat.length][mat[0].length];
		for(int i = 0; i < result.length; i++) {
			for(int j = 0; j < result[i].length; j++) {
				result[i][j] = mat[i][j] * s[j];
			}
		}
		return result;
	}
	
	public static double[][] multi(DataPoint[][] xs, double[][] w, int dimx) {
		double[][] result = new double[xs.length][];
		for(int i = 0; i < result.length; i++) {
			double[] x = SparseVector.sparseVectorToArray(xs[i], dimx);
			result[i] = multi(x, w);
		}
		return result;
	}
	
	/**
	 * 
	 */
	public static double loss(double[] a, double[] b) {
		double totle = 0;
		if(a.length != b.length) {
			return Double.NaN;
		}
		
		for(int i = 0; i < a.length; i++) {
			double sub = a[i] - b[i];
			totle += sub * sub;
		}
		return totle;
	}
	
	/**
	 * 
	 */
	public static double[] vecAbs(double[] vec) {
		double[] nvec = new double[vec.length];
		for(int i = 0; i < nvec.length; i++) {
			nvec[i] = Math.abs(vec[i]);
		}
		return nvec;
	}
	
	/**
	 * 
	 */
	public static boolean withInRange(double[] vec, double range) {
		boolean flag = true;
		for(int i = 0; i < vec.length; i++) {
			if(Math.abs(vec[i]) > range) {
				flag = false;
				break;
			}
		}
		return flag;
	}
	
	/**
	 * 
	 */
	public static double[] vecAdd(double[] a, double[] b) {
		if(a.length != b.length) {
			return null;
		}
		
		double[] result = new double[a.length];
		for(int i = 0; i < a.length; i++) {
			result[i] = a[i] + b[i];
		}
		return result;
	}
	
	/**
	 * @throws IOException 
	 * 
	 */
	public static void writeMatToFile(String filename, double[][] samples, int[][] labels) throws IOException {
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream(filename)));
		
		String line = null;
		for(int i = 0; i < labels.length; i++) {
			line = new String();
			int[] label = labels[i];
			double[] sample = samples[i];
			
			for(int j = 0; j < label.length; j++) {
				line = line + label[j];
				if(j < label.length - 1) {
					line = line + ",";
				} else {
					line = line + " ";
				}
			}
			
			for(int j = 0; j < sample.length; j++) {
				line = line + (j + 1) + ":" + sample[j] + " ";
			}
			line = line + "\n";
			out.write(line);
		}
		out.close();
	}
	
	/**
	 * 
	 */
	public static double[][][] scaleMat(double[][][] delta, double s) {
		double[][][] result = new double[delta.length][][];
		for(int i = 0; i < result.length; i++) {
			result[i] = new double[delta[i].length][];
			for(int j = 0; j < delta[i].length; j++) {
				result[i][j] = new double[delta[i][0].length];
				for(int k = 0; k < delta[i][0].length; k++) {
					result[i][j][k] = s * delta[i][j][k];
				}
			}
		} 
		return result;
	}
	
	/**
	 * 
	 */
	public static double[][][] matAdd(double[][][] a, double[][][] b) {
		double[][][] sum = new double[a.length][][];
		for(int i = 0; i < a.length; i++) {
			sum[i] = new double[a[i].length][];
			for(int j = 0; j < a[i].length; j++) {
				sum[i][j] = new double[a[i][0].length];
				for(int k = 0; k < a[i][0].length; k++) {
					sum[i][j][k] = a[i][j][k] + b[i][j][k];
				}
			}
		}
		return sum;
	}
}
