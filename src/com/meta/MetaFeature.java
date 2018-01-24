package com.meta;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import com.fileInputOutput.Problem;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;
import com.tools.Contain;
import com.tools.Sort;

public class MetaFeature {
	private Problem 	prob;
	private int[] 		ulabels;
	
	public MetaFeature(Problem prob) {
		this.prob = prob;
		this.ulabels = uniqueLabels(prob.y);
	}
	
	/**
	 * 正例个数 
	 */
	public double positiveNum(int[] labels) {
		double sum = 0;
		for(int i = 0; i < labels.length; i++) {
			if(labels[i] == 1) {
				sum = sum + 1;
			}
		}
		return sum;
	}
	
	/**
	 *  
	 */
	public int[] l2DistanceSort(Problem prob, DataPoint[] x) {
		double[] dis = new double[prob.l];
		
		for(int i = 0; i < prob.l; i++) {
			DataPoint[] sub = SparseVector.subVector(prob.x[i], x);
			double inp = SparseVector.innerProduct(sub, sub);
			double l2d = Math.pow(inp, 0.5);
			dis[i] = l2d;
		}
		
		int[] ind = Sort.getIndexBeforeSort(dis);
		return ind;
	}
	
	/**
	 * l1 distance
	 */
	public int[] l1DistanceSort(Problem prob, DataPoint[] x) {
		double[] dis = new double[prob.l];
		
		for(int i = 0; i < prob.l; i++) {
			DataPoint[] sub = SparseVector.subVector(prob.x[i], x);
			dis[i] = SparseVector.l1norm(sub); 
		}
		
		int[] ind = Sort.getIndexBeforeSort(dis);
		return ind;
	}
	
	/**
	 * 
	 */
	public int[] cosDistanceSort(Problem prob, DataPoint[] x) {
		double[] dis = new double[prob.l];
		
		for(int i = 0; i < prob.l; i++) {
			dis[i] = SparseVector.cosine(prob.x[i], x);
		}
		
		int[] ind = Sort.getIndexBeforeSort(dis);
		
		for(int i = 0; i < ind.length / 2; i++) {
			int temp = ind[i];
			ind[i] = ind[ind.length - 1 - i];
			ind[ind.length - 1 - i] = temp;
		}
		return ind;
	}
	
	/**
	 * 
	 */
	public int[] getLabels(int[][] y, int label) {
		int[] labels = new int[y.length];
		for(int i = 0; i < y.length; i++) {
			if(Contain.contain(y[i], label)) {
				labels[i] = 1;
			} else {
				labels[i] = -1;
			}
		}
		return labels;
	}
	
	/**
	 * 
	 */
	public double[] metaFeature(Problem prob, DataPoint[] x, int k, boolean tflag) {
		int[] l2index = l2DistanceSort(prob, x);		
		int[] labels = null;
		
		double[] feature = new double[this.ulabels.length * (3 * k + 2)];
		for(int i = 0; i < this.ulabels.length; i++) {
			labels = getLabels(prob.y, this.ulabels[i]);
			double[] tfeature = feature(prob, labels, x, l2index, k, tflag);
			int start = i * tfeature.length;
			for(int j = 0; j < tfeature.length; j++) {
				feature[j + start] = tfeature[j];
			}
		}
		return feature;
	}
	
	/**
	 * 转换训练集
	 */
	public double[][] transTrainSet(Problem prob, int k) {
		boolean tflag = true;
		double[][] tf = new double[prob.l][];
		DataPoint[] x = null;
		for(int i = 0; i < prob.l; i++) {
			x = prob.x[i];
			tf[i] = metaFeature(prob, x, k, tflag);
		}
		return tf;
	}
	
	/**
	 * 转换测试集
	 */
	public double[][] transTestSet(Problem train, Problem test, int k) {
		boolean tflag = false;
		double[][] tf = new double[test.l][];
		DataPoint[] x = null;
		for(int i = 0; i < test.l; i++) {
			x = test.x[i];
			tf[i] = metaFeature(train, x, k, tflag);
		}
		return tf;
	}
	
	/**
	 * 
	 */
	public DataPoint[][] trans(double[][] metaf) {
		DataPoint[][] x = new DataPoint[metaf.length][];
		DataPoint temp = null;
		for(int i = 0; i < metaf.length; i++) {
			x[i] = new DataPoint[metaf[i].length];
			for(int j = 0; j < metaf[i].length; j++) {
				int index = j + 1;
				double value = metaf[i][j];
				temp = new DataPoint(index, value);
				x[i][j] = temp;
			}
		}
		return x;
	}
	
	
	/**
	 * 
	 */
	public double[] feature(Problem prob, int[] labels, DataPoint[] x, int[] l2i, int k, boolean tflag) {
		
		DataPoint[] cm = classCenter(prob, labels);
		
		DataPoint[][] kn = getNN(prob, labels, l2i, k, tflag);
		
		double[] result = new double[3 * k + 2];
		
		int i = 0;
		for(i = 0; i < k; i++) {
			result[i] = SparseVector.distance(x, kn[i]);
		}
		
		for(i = k; i < 2 * k; i++) {
			DataPoint[] sub = SparseVector.subVector(kn[i - k], x);
			result[i] = SparseVector.l1norm(sub);
		}
		
		for(i = 2*k; i < 3*k; i++) {
			result[i] = SparseVector.cosine(x, kn[i - 2*k]);
		}
		
		result[3*k] = SparseVector.distance(x, cm);
		result[3*k+1] = SparseVector.cosine(x, cm);
		return result;
	}
	
	/**
	 * 
	 */
	public DataPoint[][] getNN(Problem prob, int[] labels, int[] index, int k, boolean flag) {
		int base = 0;
		if(flag) {
			base = 1;
		}
		
		double pn = positiveNum(labels);
		
		int min = Math.min((int)pn, k);
		
		DataPoint[][] knn = new DataPoint[k][];
		
		int i = base;
		int counter = 0;
		while(i < labels.length && counter < min) {
			if(labels[index[i]] == 1) {
				knn[counter++] = prob.x[index[i]];
			}
			i++;
		}
		
		while(counter < k) {
			knn[counter] = knn[counter - 1];
			counter++;
		}
		
		return knn;
	}
	/**
	 * 
	 */
	public DataPoint[] classCenter(Problem prob, int[] labels) {
		DataPoint[] sum = null;
		for(int i = 0; i < prob.l; i++) {
			if(labels[i] == 1) {
				sum = SparseVector.addVector(sum, prob.x[i]);
			}
		}
		
		double scale = 1 / positiveNum(labels);
		SparseVector.scaleVector(sum, scale);
		return sum;
	}
	
	/**
	 * 
	 */
	public int[] uniqueLabels(int[][] y) {
		Set<Integer> set = new HashSet<Integer>();
		for(int i = 0; i < y.length; i++) {
			for(int j = 0; j < y[i].length; j++) {
				set.add(y[i][j]);
			}
		}
		
		int[] labels = new int[set.size()];
		Iterator<Integer> it = set.iterator();
		int counter = 0;
		while(it.hasNext()) {
			labels[counter++] = it.next();
		}
		return labels;
	}
}
