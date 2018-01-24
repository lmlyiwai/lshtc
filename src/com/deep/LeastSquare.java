package com.deep;

import java.util.Random;

import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;
import com.tools.RandomSequence;
import com.tools.Sort;

public class LeastSquare {
	/**
	 * ���
	 */
	public static double[][] solve(double[][] X, double[][] Y, double lr, double precision, double epoch) {
		int wr = X[0].length;
		int wc = Y[0].length;
		int n = X.length;
		
		double[][] w = new double[wr][wc];
		Matrix.randInitMat(w, -0.5, 0.5);
		
		int ct = 0;
		
		double lastObj = Double.POSITIVE_INFINITY;
		while(ct < epoch) {
			int[] index = RandomSequence.randomSequence(n);
			for(int i = 0; i < n; i++) {
				double[] x = X[index[i]];
				double[] y = Y[index[i]];
				double[] it2 = Matrix.multi(x, w);
				double[] it3 = Matrix.vectorSub(it2, y);
				double[][] delta = Matrix.multi(x, it3);
				double scale = -2 * lr;
				delta = Matrix.scale(delta, scale);
				w = Matrix.matrixAdd(w, delta);
			}
			
			double sum = 0;
			for(int i = 0; i < n; i++) {
				double[] x = X[i];
				double[] y = Y[i];
				double[] sub = Matrix.vectorSub(Matrix.multi(x, w), y);
				sum = sum + Matrix.innerProcuct(sub, sub);
			}
			
			if(Math.abs(sum - lastObj) / lastObj < precision) {
				break;
			}
			
			lastObj = sum;
			ct = ct + 1;
		}
		return w;
	}
	
	/**
	 * 
	 */
	public static double[][] solve(DataPoint[][] X, double[][] Y, int dim, double lr, double precision, double epoch) {
		int wr = dim;
		int wc = Y[0].length;
		int n = X.length;
		
		double[][] w = new double[wr][wc];
		Matrix.randInitMat(w, -0.5, 0.5);
		
		int ct = 0;
		
		double lastObj = Double.POSITIVE_INFINITY;
		while(ct < epoch) {
			int[] index = RandomSequence.randomSequence(n);
			for(int i = 0; i < n; i++) {
				DataPoint[] tx = X[index[i]];
				double[] x = SparseVector.sparseVectorToArray(tx, dim);
				double[] y = Y[index[i]];
				double[] it2 = Matrix.multi(x, w);
				double[] it3 = Matrix.vectorSub(it2, y);
				double[][] delta = Matrix.multi(x, it3);
				double scale = -2 * lr;
				delta = Matrix.scale(delta, scale);
				w = Matrix.matrixAdd(w, delta);
			}
			
			double sum = 0;
			for(int i = 0; i < n; i++) {
				DataPoint[] tx = X[i];
				double[] x = SparseVector.sparseVectorToArray(tx, dim);
				double[] y = Y[i];
				double[] sub = Matrix.vectorSub(Matrix.multi(x, w), y);
				sum = sum + Matrix.innerProcuct(sub, sub);
			}
			if(Math.abs(sum - lastObj) / lastObj < precision) {
				break;
			}
			
			lastObj = sum;
			ct = ct + 1;
		}
		return w;
	}
	
	/**
	 * ���
	 */
	public static double[][] newSolve(double[][] X, double[][] Y, double lr, double precision, double epoch) {
		int wr = X[0].length;
		int wc = Y[0].length;
		int n = X.length;
		
		double[][] w = new double[wr][wc];
		Matrix.randInitMat(w, -0.5, 0.5);
		w = initW(X, w, 1.3863);
		
		int ct = 0;
		
		double lastObj = Double.POSITIVE_INFINITY;
		while(ct < epoch) {
			int[] index = RandomSequence.randomSequence(n);
			for(int i = 0; i < n; i++) {
				double[] x = X[index[i]];
				double[] y = Y[index[i]];
				double[] it2 = Matrix.multi(x, w);
				double[] it3 = Matrix.vectorSub(it2, y);
				double[][] delta = Matrix.multi(x, it3);
				double scale = -2 * lr;
				delta = Matrix.scale(delta, scale);
				
				deltaW(x, w, delta, 2.1972);              //ȷ�����ֵ��ָ����Χ��
				
				w = Matrix.matrixAdd(w, delta);
			}
			
			double sum = 0;
			for(int i = 0; i < n; i++) {
				double[] x = X[i];
				double[] y = Y[i];
				double[] sub = Matrix.vectorSub(Matrix.multi(x, w), y);
				sum = sum + Matrix.innerProcuct(sub, sub);
			}
			
			if(Math.abs(sum - lastObj) / lastObj < precision) {
				break;
			}
			
			lastObj = sum;
			ct = ct + 1;
		}
		return w;
	}
	
	/**
	 * ��ʼ��Ȩֵ��ʹ�������ָ����Χ�࣬���¹����б�֤������ָ����Χ
	 */
	public static double[][] newSolve(DataPoint[][] X, double[][] Y, int dim, double lr, double precision, double epoch) {
		int wr = dim;
		int wc = Y[0].length;
		int n = X.length;
		
		double[][] w = new double[wr][wc];
		Matrix.randInitMat(w, -0.5, 0.5);
		
		w = initW(X, w, 1.3863);               //
		
		int ct = 0;
		
		double lastObj = Double.POSITIVE_INFINITY;
		while(ct < epoch) {
			int[] index = RandomSequence.randomSequence(n);
			long start = System.currentTimeMillis();
			for(int i = 0; i < n; i++) {
				DataPoint[] x = X[index[i]];
				double[] y = Y[index[i]];
				double[] it2 = Matrix.multi(x, w);                //ϡ��������˿��ܸ���
				double[] it3 = Matrix.vectorSub(it2, y);
				double[] fullx = SparseVector.sparseVectorToArray(x, dim); 
//				double[][] delta = Matrix.multi(fullx, it3);
				double[][] delta = Matrix.sparseMulti(fullx, it3);
				double scale = -2 * lr;
//				delta = Matrix.scale(delta, scale);                //ԭ�ظı� 1
//				Matrix.localScale(delta, scale);
				Matrix.sparseLocalScale(delta, scale);
				
//				deltaW(fullx, w, delta, 2.1972);              //ȷ�����ֵ��ָ����Χ�� 2
				
//				w = Matrix.matrixAdd(w, delta);                  //�����ʱ
//				Matrix.matrixLocalAdd(w, delta);
				Matrix.sparseMatAdd(w, delta);
			}
			
			double sum = 0;
			for(int i = 0; i < n; i++) {
				DataPoint[] tx = X[i];
				double[] y = Y[i];
				double[] sub = Matrix.vectorSub(Matrix.multi(tx, w), y);
				sum = sum + Matrix.innerProcuct(sub, sub);
			}
			if(Math.abs(sum - lastObj) / lastObj < precision) {
				break;
			}
			long end = System.currentTimeMillis();
			System.out.println(sum + ", " + (end - start) + "ms");
			lastObj = sum;
			ct = ct + 1;
		}
		return w;
	}
	
	/**
	 * �仯Ȩֵ������ʹ��������������ڸ�����Χ��
	 */
	public static double[][] initW(DataPoint[][] X, double[][] w, double range) {
		boolean flag = true;
		double[][] result = new double[w.length][w[0].length];        //����wԭ�ظı�
		Matrix.copyMat(w, result);
		
		int counter = 0;
		while(true) {
			counter = 0;
			double[][] y = Matrix.multi(X, result);
			double[] colm = Matrix.colMax(y);
			for(int i = 0; i < colm.length; i++) {
				if(colm[i] > range) {
					counter++;
					colm[i] = 0.6;
				} else {
					colm[i] = 1;
				}
			}
			
			if(counter == 0) {
				flag = true;
			} else {
				flag = false;
			}
			
			if(flag) {
				break;
			}
			result = Matrix.scale(result, colm);              //����ԭ�ظı�
		}
		return result;
	}
	
	/**
	 * �仯Ȩֵ������ʹ��������������ڸ�����Χ��
	 */
	public static double[][] initW(double[][] X, double[][] w, double range) {
		boolean flag = true;
		double[][] result = new double[w.length][w[0].length];
		Matrix.copyMat(w, result);
		
		int counter = 0;
		while(true) {
			counter = 0;
			double[][] y = Matrix.multi(X, result);
			double[] colm = Matrix.colMax(y);
			for(int i = 0; i < colm.length; i++) {
				if(colm[i] > range) {
					counter++;
					colm[i] = 0.6;
				} else {
					colm[i] = 1;
				}
			}
			
			if(counter == 0) {
				flag = true;
			} else {
				flag = false;
			}
			
			if(flag) {
				break;
			}
			result = Matrix.scale(result, colm);
		}
		return result;
	}
	
	/**
	 * 
	 */
	public static void deltaW(double[] x, double[][] w, double[][] dw, double range) {
		boolean flag = false;
		double[] ty = Matrix.multi(x, w);
		flag = Matrix.withInRange(ty, range);
		if(!flag) {
			for(int i = 0; i < dw.length; i++) {
				for(int j = 0; j < dw[i].length; j++) {
					dw[i][j] = 0;
				}
			}
		}
		
		while(true) {
			double[][] tw = Matrix.matrixAdd(w, dw);	
		    ty = Matrix.multi(x, tw);
			flag = Matrix.withInRange(ty, range);
			if(flag) {
				break;
			} else {
				Matrix.localScale(dw, 0.5);
			}
		}
	}
}
