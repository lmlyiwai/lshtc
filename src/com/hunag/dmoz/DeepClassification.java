/**
 * Deep Classification in Large-scale Text Hierarchies
 *  Gui-Rong Xue
 *  SIGIR'08, July 20-24, 2008, Singapore
 *  实现为单类标分类问题
 */
package com.hunag.dmoz;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import com.fileInputOutput.Problem;
import com.refinedExpert.Tools;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;
import com.tools.Sort;

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
	
	/**
	 * 简略预测，TopN个标签中是否包含真实类别
	 */
	public void predict(Problem test, int[] ks) {
		if (test == null) {
			return;
		}
		
		double[] count = new double[ks.length];
		for (int i = 0; i < test.l; i++) {
System.out.print(i);
long start = System.currentTimeMillis();
			normalizeTF(test.x[i]);
			double[] cos = new double[this.classCenters.length];
			for (int j = 0; j < this.classCenters.length; j++) {
				cos[j] = cosine(test.x[i], this.classCenters[j]);
			}
			int[] index = Sort.getIndexBeforeSort(cos);
			for (int m = 0; m < ks.length; m++) {
				int[] topnlabel = new int[ks[m]];
				int base = index.length - ks[m];
				for (int j = base; j < index.length; j++) {
					topnlabel[j-base] = this.ulabels[index[j]];
				}
				if (Tools.isContain(topnlabel, test.y[i][0])) {
					count[m] += 1;
				}
			}
long end = System.currentTimeMillis();
System.out.println(", time = " + (end - start));
		}
		for (int i = 0; i < ks.length; i++) {
			System.out.println("n = " + ks[i] + ", accuracy = " + (count[i] / test.l));
		}
	}

	/**
	 * 简略预测，TopN个标签中是否包含真实类别
	 * @throws IOException 
	 */
	public void predict(Problem test) throws IOException {
		if (test == null) {
			return;
		}
		
		double count = 0;
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		for (int i = 0; i < test.l; i++) {
System.out.print(i);
long start = System.currentTimeMillis();
			normalizeTF(test.x[i]);
			double[] cos = new double[this.classCenters.length];
			for (int j = 0; j < this.classCenters.length; j++) {
				cos[j] = cosine(test.x[i], this.classCenters[j]);
			}
			int[] index = Sort.getIndexBeforeSort(cos);
			for (int j = 0; j < index.length; j++) {
				int idx = index.length - 1 - j;
				if (this.ulabels[index[idx]] == test.y[i][0]) {
					count += j;
					addMap(map, (j+1));
					System.out.println(", sort = " + (j+1));
					break;
				}
			}
long end = System.currentTimeMillis();
		}
		System.out.println("Average rank " + (count / test.l));
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream("statics" + this.topn + ".txt"))) ;
		Set<Integer> set = map.keySet();
		Iterator<Integer> it = set.iterator();
		while (it.hasNext()) {
			int key = it.next();
			int value = map.get(key);
			out.write(key + ":" + value + "\n");
		}
		out.close();
	}
	
	public int getTopn() {
		return topn;
	}

	public void setTopn(int topn) {
		this.topn = topn;
	}
	
	private void addMap(Map<Integer, Integer> map, int key) {
		if (map == null) {
			return;
		}
		if (map.containsKey(key)) {
			int value = map.get(key);
			value = value + 1;
			map.put(key, value);
		} else {
			map.put(key, 1);
		}
	}
	
	/**
	 * 统计样本集中每个类样本数目
	 * @throws IOException 
	 */
	public void statisticNumOfeachClass(Problem prob, String outfile) throws IOException {
		if (prob == null) {
			return;
		}
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		for (int i = 0; i < prob.l; i++) {
			for (int j = 0; j < prob.y[i].length; j++) {
				addMap(map, prob.y[i][j]);
			}
		}
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(outfile)));
		Set<Integer> set = map.keySet();
		Iterator<Integer> it = set.iterator();
		while (it.hasNext()) {
			int key = it.next();
			int value = map.get(key);
			out.write(key + ":" + value + "\n");
		}
		out.close();
	}
	
	/**
	 * @param extend为true时给每个样本添加一维偏置
	 */
	public void transformSamples(Problem prob, boolean extend) {
		if (prob == null) {
			return;
		}
		for (int i = 0; i < prob.l; i++) {
			normalizeTF(prob.x[i]);
			double[] cos = new double[this.classCenters.length];
			for (int j = 0; j < this.classCenters.length; j++) {
				cos[j] = cosine(this.classCenters[j], prob.x[i]);
			}
			int[] idx = Sort.getIndexBeforeSort(cos);
			int dim = this.topn;
			if (extend) {
				dim = dim + 1;
			}
			DataPoint[] dp = new DataPoint[dim];
			for (int j = 0; j < this.topn; j++) {
				int index = idx[idx.length - 1 - j] + 1;
				double value = cos[idx[idx.length - 1 - j]];
				dp[j] = new DataPoint(index, value);
			}
			dp[dim-1] = new DataPoint(this.classCenters.length + 1, 1);
			prob.x[i] = dp;
		}
	}
}
