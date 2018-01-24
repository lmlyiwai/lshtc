package com.knn;

import java.util.HashMap;
import java.util.Map;

import com.fileInputOutput.Problem;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;
import com.tools.Contain;
import com.tools.ProcessProblem;
import com.tools.Sort;

//Ml-KNN:A lazy learning approach to multi-label learning
public class MLKNN {
	private Problem 		prob;
	private int[] 			labels;
	private double[][] 		prior;
	private double[][][] 	post;
	private int 			k;
	
	public MLKNN(Problem prob) {
		this.prob = prob;
		this.labels = ProcessProblem.getUniqueLabels(prob.y);
	}
	
	/**
	 * 统计先验概率，后验概率等 
	 * prob 训练集
	 * K  近邻数
	 * s  平滑参数
	 */
	public void getStatistic(Problem prob, int K, double s) {
		this.k = K;
		
		double[][] statistic = new double[2][this.labels.length];
		
		//计算先验概率
		for(int i = 0; i < this.labels.length; i++) {
			int label = this.labels[i];
			double counter = getLabelCount(prob, label);
			statistic[0][i] = (s + counter) / (s * 2 + prob.l);
			statistic[1][i] = 1 - statistic[0][i];
		}
		
		//
		
		double[][][] labelkbinary = new double[this.labels.length][K+1][2];
		
		for(int i = 0; i < prob.l; i++) {
			DataPoint[] x = prob.x[i];
			int[][] klabel = getKnnLabels(prob, x, K);
			int[] labelCount = getCountVector(klabel);
			for(int j = 0; j < labelCount.length; j++) {
				int label = this.labels[j];
				if(Contain.contain(prob.y[i], label)) {
					labelkbinary[j][labelCount[j]][0]++;
				} else {
					labelkbinary[j][labelCount[j]][1]++;
				}
			}
			
		}
		
		for(int i = 0; i < labelkbinary.length; i++) {
			double[][] p = labelkbinary[i];
			double sump = 0;
			double sumn = 0;
			for(int j = 0; j < p.length; j++) {
				sump += p[j][0];
				sumn += p[j][1];
			}
			for(int j = 0; j < p.length; j++) {
				p[j][0] = (s + p[j][0]) / (s * (K + 1) + sump);
				p[j][1] = (s + p[j][1]) / (s * (K + 1) + sumn);
			}
		}
		
		this.prior = statistic;
		this.post = labelkbinary;
	}
	
	/**
	 * 获得样本中类标label出现的次数 
	 */
	public double getLabelCount(Problem prob, int label) {
		double counter = 0;
		for(int i = 0; i < prob.l; i++) {
			if(Contain.contain(prob.y[i], label)) {
				counter++;
			} 
		}
		return counter;
	}
	
	/**
	 * 获得样本i的K个近邻类标 
	 */
	public int[][] getKnnLabels(Problem prob, DataPoint[] x, int k) {
		double[] distance = new double[prob.l];
		
		for(int j = 0; j < prob.l; j++) {
			DataPoint[] sub = SparseVector.subVector(prob.x[j], x);
			double sq = SparseVector.innerProduct(sub, sub);
			double dis = Math.pow(sq, 0.5);
			distance[j] = dis;
		}
		
		int[] kindex = Sort.getIndexBeforeSort(distance);
		
		int[][] result = new int[k][];
		for(int j = 0; j < k; j++) {
			result[j] = prob.y[kindex[j+1]];
		}
		return result;
	}
	
	/**
	 * 类标数组变为出现出现次数向量 
	 */
	public int[] getCountVector(int[][] y) {
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		int key;
		int value;
		for(int i = 0; i < y.length; i++) {
			for(int j = 0; j < y[i].length; j++) {
				key = y[i][j];
				if(map.containsKey(key)) {
					value = map.get(key);
					value = value + 1;
					map.put(key, value);
				} else {
					map.put(key, 1);
				}
			}
		}
		
		int[] result = new int[this.labels.length]; 
		for(int i = 0; i < result.length; i++) {
			if(map.get(this.labels[i]) != null) {
				result[i] = map.get(this.labels[i]);
			} else {
				result[i] = 0;
			}
		}
		return result;
	}
	
	/**
	 * 预测单个样本
	 */
	public int[] predictSingle(Problem prob, DataPoint[] x) {
		
		int[][] neary = getKnnLabels(prob, x, this.k);
		int[] vy = getCountVector(neary);
		
		int[] result = new int[this.labels.length];
		for(int i = 0; i < this.labels.length; i++) {
			double p = this.prior[0][i] * this.post[i][vy[i]][0];
			double n = this.prior[1][i] * this.post[i][vy[i]][1];
			if(p > n) {
				result[i] = 1;
			} else if(p == n) {
				result[i] = (int)Math.round(Math.random());
			} else {
				result[i] = 0;
			}
		}
		
		int count = 0;
		for(int i = 0; i < result.length; i++) {
			if(result[i] == 1) {
				count++;
			}
		}
		
		int[] rl = new int[count];
		count = 0;
		for(int i = 0; i < result.length; i++) {
			if(result[i] == 1) {
				rl[count++] = this.labels[i];
			}
		}
		return rl;
	}

	/**
	 *  
	 */
	public int[][] predict(Problem prob, DataPoint[][] xs) {
		int[][] y = new int[xs.length][];
		DataPoint[] x = null;
		for(int i = 0; i < xs.length; i++) {
			x = xs[i];
			y[i] = predictSingle(prob, x);
		}
		return y;
	}

	public int[] getLabels() {
		return labels;
	}

	public void setLabels(int[] labels) {
		this.labels = labels;
	}

	public int getK() {
		return k;
	}

	public void setK(int k) {
		this.k = k;
	}
	
	
}
