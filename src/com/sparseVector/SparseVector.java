package com.sparseVector;

import java.util.ArrayList;
import java.util.List;

/**
 * 稀疏矩阵求和，差，内积等操作
 **/
public class SparseVector {
	
	/**
	 * 两向量求内积
	 */
	public static double innerProduct(DataPoint[] a, DataPoint[] b) {
		double result = Double.NEGATIVE_INFINITY;
		
		if(a == null || b == null) {
			return result;
		}
		
		int pa = 0;
		int pb = 0;
		
		int indexa;
		int indexb;
		
		result = 0;
		while(pa < a.length && pb < b.length) {
			indexa = a[pa].index;
			indexb = b[pb].index;
			
			if(indexa == indexb) {
				result += a[pa].value * b[pb].value;
				pa++;
				pb++;
			} else if(indexa < indexb) {
				pa++;
			} else {
				pb++;
			}
		}
		return result;
	}
	
	public static double innerProduct(DataPoint[] a, DataPoint[] b, int n) {
		double[] fulla = sparseVectorToArray(a, n);
		double[] fullb = sparseVectorToArray(b, n);
			
		double r = 0;
		for(int i = 0; i < fulla.length; i++) {
			r += fulla[i] * fullb[i];
		}
		
		return r;
	}

	/**
	 * 对输入向量扩大scale倍，输入向量自身改变
	 **/
	
	public static void scaleVector(DataPoint[] a, double scale) {
		if(a != null) {
			int pa = 0;
			while(pa < a.length) {
				a[pa].value = a[pa].value * scale;
				pa++;
			}
		}
	}

	/**
	 * 
	 */
	public static DataPoint[] slVector(DataPoint[] a, double scale) {
		if(a != null) {
			DataPoint[] result = new DataPoint[a.length];
			int pa = 0;
			while(pa < a.length) {
				result[pa] = new DataPoint(a[pa].index, a[pa].value * scale);
				pa++;
			}
			return result;
		}
		return null;
	}
	
	/**
	 * 向量相加，输入向量保持不变
	 * 
	 * */
	public static DataPoint[] addVector(DataPoint[] a, DataPoint[] b) {
		if(a == null  && b == null) {
			return null;
		}
		
		int lengtha = 0;
		int lengthb = 0;
		
		if(a == null && b != null) {
			lengtha = 0;
			lengthb = b.length;
		} else if(a != null && b == null) {
			lengtha = a.length;
			lengthb = 0;
		} else {
			lengtha = a.length;
			lengthb = b.length;
		}
		
		
		ArrayList<DataPoint> list = new ArrayList<DataPoint>();
		
		int pa = 0;
		int pb = 0;
		
		int indexa = 0;
		int indexb = 0;
		
		double sum;
		
		DataPoint temp;
		while(pa < lengtha && pb < lengthb) {
			
			indexa = a[pa].index;
			indexb = b[pb].index;
			
			if(indexa == indexb) {
				sum = a[pa].value + b[pb].value;
				temp = new DataPoint(indexa, sum);
				if(sum != 0) {
					list.add(temp);
				}
				pa++;
				pb++;
			} else if(indexa < indexb) {
				sum = a[pa].value;
				temp = new DataPoint(indexa, sum);
				if(sum != 0) {
					list.add(temp);
				}
				pa++;
			} else {
				sum = b[pb].value;
				temp = new DataPoint(indexb, sum);
				if(sum != 0) {
					list.add(temp);
				}
				pb++;
			}
		}
		
		while(pa < lengtha) {
			indexa = a[pa].index;
			sum = a[pa].value;
			temp = new DataPoint(indexa, sum);
			if(sum != 0) {
				list.add(temp);
			}
			pa++;
		}
		
		while(pb < lengthb) {
			indexb = b[pb].index;
			sum = b[pb].value;
			temp = new DataPoint(indexb, sum);
			if(sum != 0) {
				list.add(temp);
			}
			pb++;
		}
		
		DataPoint[] result = new DataPoint[list.size()];
		for(int i = 0; i < result.length; i++) {
			result[i] = list.get(i);
		}
		return result;
	} 

	
	public static DataPoint[] addVector(DataPoint[] a, DataPoint[] b, int n) {
		double[] result = new double[n];
		for(int i = 0; i < result.length; i++) {
			result[i] = 0;
		}
		
		if(a != null) {
			for(DataPoint d : a) {
				result[d.index - 1] = d.value;
			}
		}
		
		if(b != null) {
			for(DataPoint d : b) {
				result[d.index - 1] += d.value;
			}
		}
		
		return arrayToSparseVector(result);
	}
	
	/**
	 *  向量相减 a - b
	 * */
	public static DataPoint[] subVector(DataPoint[] a, DataPoint[] b) {
		if(a == null && b == null) {
			return null;
		}
		
		if(a != null && b == null) {
			return SparseVector.copyScaleVector(a, 1);
		} else if(a == null && b != null) {
			return SparseVector.copyScaleVector(b, -1);
		} else {
			DataPoint[] sub = SparseVector.copyScaleVector(b, -1);
			return SparseVector.addVector(a, sub); 
		}
		
	}
	
	
	public static DataPoint[] subVector(DataPoint[] a, DataPoint[] b, int n) {
		double[] result = new double[n];
		for(int i = 0; i < result.length; i++) {
			result[i] = 0;
		}
		
		if(a != null) {
			for(DataPoint d : a) {
				result[d.index - 1] = d.value;
			}
		}
		
		if(b != null) {
			for(DataPoint d : b) {
				result[d.index - 1] -= d.value;
			}
		}
		
		return arrayToSparseVector(result);
	}
	
	/**
	 *  向量复制，并且做缩放。复制是深复制，返回一个新的向量。
	 * */
	public static DataPoint[] copyScaleVector(DataPoint[] a, double scale) {
		if(a == null) {
			return null;
		} 
		
		DataPoint[] result = new DataPoint[a.length];
		for(int i = 0; i < a.length; i++) {
			result[i] = new DataPoint(a[i].index, a[i].value * scale);
		}
		return result;
	}
	
	/**
	 * 打印向量
	 * */
	public static void showVector(DataPoint[] a) {
		if(a == null) {
			return;
		}
		
		for(int i = 0; i < a.length; i++) {
			System.out.print(a[i].index + ":" + a[i].value);
			if(i < a.length - 1) {
				System.out.print(" ");
			} else {
				System.out.println();
			}
		}
	}
	
	/**
	 * 全向量转换为稀疏向量
	 * */
	public static DataPoint[] arrayToSparseVector(double[] array) {
		if(array == null) {
			return null;
		}
		
		List<DataPoint> list = new ArrayList<DataPoint>();
		for(int i = 0; i < array.length; i++) {
			if(array[i] != 0) {
				list.add(new DataPoint(i+1, array[i]));
			}
		}
		
		DataPoint[] result = new DataPoint[list.size()];
		for(int i = 0; i < result.length; i++) {
			result[i] = list.get(i);
		}
		return result;
	}
	
	/**
	 * 稀疏向量转化为全向量
	 * */
	public static double[] sparseVectorToArray(DataPoint[] dp, int n) {
		if(dp == null || dp.length == 0) {
			return null;
		}
		
		double[] result = new double[n];
		
		for(int i = 0; i < result.length; i++) {
			result[i] = 0.0;
		}
		
		for(DataPoint d : dp) {
			result[d.index - 1] = d.value;
		}
		
		return result;
	}
	
	public static double distance(int[] a, int[] b) {
		if(a == null || b == null || a.length != b.length) {
			System.out.println("error in Sparse Vector distance.");
			return Double.NaN;
		}
		
		double sum = 0;
		for(int i = 0; i < a.length; i++) {
			sum += Math.abs(a[i] - b[i]);
		}
		
		return sum;
	}
	
	/**
	 * y中大于thresholds的输出1,小于thresholds的输出-1
	 * */
	public static int[] cutLabels(double[] y, double t) {
		if(y == null) {
			return null;
		}
		
		int[] result = new int[y.length];
		for(int i = 0; i < y.length; i++) {
			if(y[i] >= t) {
				result[i] = 1;
			} else {
				result[i] = -1;
			}
		}
		
		return result;
	}
	
	public static double maxRecall(int[] y, double[] pre) {
		double min = Double.MAX_VALUE;
		for(int i = 0; i < y.length; i++) {
			if(y[i] == 1) {
				if(pre[i] < min) {
					min = pre[i];
				}
			}
		}
		return min;
	}
	
	public static double innerProduct(double[] w, DataPoint[] dp) {
		if(w == null || dp == null) {
//			System.out.println("In sparse vector, inner product, w and dp cant't be null.");
			return Double.NEGATIVE_INFINITY;
		}
		double result = 0.0;
		int n = w.length;
		for(int i = 0; i < dp.length; i++) {
			if(dp[i].index <= n) {
				result += w[dp[i].index - 1] * dp[i].value;
			}
		}
		return result;
	}
	
	/**
	 * 
	 */
	public static double innerProduct(double[] a, double[] b) {
		if(a == null || b == null) {
			return Double.NEGATIVE_INFINITY;
		}
		
		int la = a.length;
		int lb = b.length;
		int l = la < lb ? la : lb;
		double inp = 0;
		for(int i = 0; i < l; i++) {
			inp += a[i] * b[i];
		}
		return inp;
	}
	
	/**
	 * 
	 */
	public static double[] addVector(double[] a, double[] b) {
		if(a.length != b.length) {
			return null;
		}
		
		double[] result = new double[a.length];
		for(int i = 0; i< result.length; i++) {
			result[i] = a[i] + b[i];
		}
		return result;
	}
	
	/**
	 * 
	 */
	public static double[] subVector(double[] a, double[] b) {
		if(a == null || b == null || a.length != b.length) {
			return null;
		}
		
		double[] result = new double[a.length];
		for(int i = 0; i < result.length; i++) {
			result[i] = a[i] - b[i];
		}
		return result;
	}
	
	/**
	 * 
	 */
	public static double l1norm(DataPoint[] x) {
		double sum = 0;
		for(int i = 0; i < x.length; i++) {
			sum = sum + Math.abs(x[i].value);
		}
		return sum;
	}
	
	/**
	 * 
	 */
	public static double cosine(DataPoint[] a, DataPoint[] b) {
		double inpa = innerProduct(a, a);
		double l2a = Math.pow(inpa, 0.5);
		
		double inpb = innerProduct(b, b);
		double l2b = Math.pow(inpb, 0.5);
		
		double cos = SparseVector.innerProduct(a, b) / (l2a * l2b);
		return cos;
	}
	
	/**
	 * 
	 */
	public static double distance(DataPoint[] a, DataPoint[] b) {
		DataPoint[] sub = SparseVector.subVector(a, b);
		double inp = SparseVector.innerProduct(sub, sub);
		double norm = Math.pow(inp, 0.5);
		return norm;
	}
	
	/**
	 * 
	 */
	public static double[] subVector(DataPoint[] a, double[] b) {
		double[] result = new double[b.length];
		for(int i = 0; i < result.length; i++) {
			result[i] = -b[i];
		}
		
		for(int i = 0; i < a.length; i++) {
			int index = a[i].index - 1;
			double value = a[i].value;
			result[index] += value;
		}
		return result;
	}
	
	/**
	 * 
	 */
	public static double[] scaleVector(double[] v, double s) {
		double[] result = new double[v.length];
		for(int i = 0; i < result.length; i++) {
			result[i] = v[i] * s;
		}
		return result;
	}
	
	/**
	 * a + b, 和保存在a中
	 */
	public static void localVecAdd(double[] a, double[] b) {
		if(a.length != b.length) {
			return;
		}
		
		for(int i = 0; i < a.length; i++) {
			a[i] += b[i];
		}
	}
	
	/**
	 * 
	 */
	public static void vectorAdd(double[] a, double[] b) {
		if(a.length != b.length) {
			return;
		}
		
		for(int i = 0; i < a.length; i++) {
			a[i] += b[i];
		}
	}
	
	/**
	 * 
	 */
	public static void localVecScale(double[] v, double s) {
		for(int i = 0; i < v.length; i++) {
			v[i] *= s;
		}
	}
}
