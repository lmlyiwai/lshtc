/**
 * Deep Classification in Large-scale Text Hierarchies
 *  Gui-Rong Xue
 *  SIGIR'08, July 20-24, 2008, Singapore
 *  实现为单类标分类问题
 */
package com.hunag.dmoz;

import java.util.Map;

import com.fileInputOutput.Problem;
import com.refinedExpert.Tools;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;

public class DeepClassification {
	private Problem train;
	private int topn;
	private int[] ulabels;
	private DataPoint[][] classCenters;
	
	public DeepClassification(Problem train, int topn) {
		this.train = train;
		this.topn = topn;
		this.ulabels = Tools.getUniqueItem(this.train.y);
		this.classCenters = new DataPoint[this.ulabels.length][];
	}
	
	/**
	 *计算每一类的样本中心，以正则化的 term frequency 表示
	 *@reurn 各类中心，this.classCenters
	 */
	public void getClassCenter() {
		if (this.train == null || this.ulabels == null) {
			return;
		}
		for (int i = 0; i < this.ulabels.length; i++) {
			int label = this.ulabels[i];
System.out.println(label);
			DataPoint[] center = null;
			for (int j = 0; j < this.train.l; j++) {
				if (this.train.y[j][0] == label) {
					center = SparseVector.addVector(center, this.train.x[j]);
				}
			}
			normalizeTF(center);
			this.classCenters[i] = center;
		}
	}
	
	/**
	 * 
	 */
	private double findMaxValue(DataPoint[] dp) {
		if (dp == null) {
			return Double.NaN;
		}
		double max = Double.NEGATIVE_INFINITY;
		for (DataPoint d : dp) {
			if (d.value > max) {
				max = d.value;
			}
		}
		return max;
	}
	
	/**
	 * 
	 */
	private void normalizeTF(DataPoint[] dp) {
		if (dp == null) {
			return;
		}
		double max = findMaxValue(dp);
		for (DataPoint d : dp) {
			d.value = d.value / max;
		}
	}
	
	/**
	 * @param x, y稀疏向量
	 * @return x, y的cosine夹角
	 */
	private double cosine(DataPoint[] x, DataPoint[] y) {
		if (x == null || y == null) {
			return Double.NaN;
		}
		
		double innerp = SparseVector.innerProduct(x, y);
		double normx = SparseVector.innerProduct(x, x);
		normx = Math.sqrt(normx);
		double normy = SparseVector.innerProduct(y, y);
		normy = Math.sqrt(normy);
		return innerp / (normx * normy);
	}
	
	/**
	 * multi-variate bernouli model
	 *  
	 */
	public void getBernouliTheta() {
		if (this.train == null) {
			return;
		}
		
		double[] countOfEachClass = new double[this.ulabels.length];
		for (int i = 0; i < this.ulabels.length; i++) {
			int label = this.ulabels[i];
			
			double numOfClassi = 0;
			for (int j = 0; j < this.train.l; j++) {
				if (Tools.isContain(this.train.y[j], label)) {
					numOfClassi++;
				}
			}
			
			
		}
	}
	
	/**
	 * 计算Bernouli Model
	 */
	private Map<Integer, Double> getBernouliClassTheta(int label, double numOfClass) {
		return null;
	}
}
