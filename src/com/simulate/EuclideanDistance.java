package com.simulate;

import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Linear;
import com.rssvm.Measures;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;
import com.tools.Contain;
import com.tools.ProcessProblem;

public class EuclideanDistance {
	private Problem 		prob;
	private int[] 			ulabels;
	private DataPoint[][] 	weight;
	
	public EuclideanDistance(Problem prob) {
		this.prob = prob;
		this.ulabels = ProcessProblem.getUniqueLabels(prob.y);
	}
	
	/**
	 * 
	 */
	public DataPoint[][] train(Problem prob, double lr, double epoch, double lamda) {
		double[] cs = new double[15];
		for(int i = 0; i < cs.length; i++) {
			cs[i] = Math.pow(2, i - 7);
		}
		DataPoint[][] weight = initWeight(prob, cs);
		
		for(int i = 0; i < this.ulabels.length; i++) {
			int label = this.ulabels[i];
			double[] obj = object(prob.y, label);
			weight[i] = updateWeight(prob.x, obj, 0.001, 1000, weight[i], lamda);
		}
		this.weight = weight;
		return weight;
	}
	
	/**
	 * 更新单个权值 
	 */
	public DataPoint[] updateWeight(DataPoint[][] X, double[] obj, double lr, double epoch, DataPoint[] w_old, double lambda) {
		double[] w = SparseVector.sparseVectorToArray(w_old, this.prob.n);
		int tc = 0;
		int index = 0;
		DataPoint[] x = null;
		double d = 0;
		double o = 0;
		double delta = 0;
		double delta1 = 0;
		double[] deltaW = null;
		while(tc++ < epoch * X.length) {
			index = (int)Math.floor((Math.random() * X.length));
			x = X[index];            //随机选取样本                             
			d = obj[index];          // 期望输出
			
			o = sigmoid(SparseVector.innerProduct(w, x));
			delta = 2 * lr * (d - o) * o * (1 - o);
			delta1 = lr * lambda;
			
			deltaW = SparseVector.subVector(SparseVector.copyScaleVector(x, delta), SparseVector.scaleVector(w, delta1));
			
			SparseVector.localVecAdd(w, deltaW);
		}
		
		return SparseVector.arrayToSparseVector(w);
	}
	
	/**
	 * 支持向量机初始化权值 
	 */
	public DataPoint[][] initWeight(Problem prob, double[] cs) {
		int[][] index = Tools.getClassIndex(prob.y);
		int[][] tvindex = Tools.splits(index);
		
		Problem train = new Problem();
		train.bias = prob.bias;
		train.l = tvindex[0].length;
		train.n = prob.n;
		train.x = new DataPoint[train.l][];
		train.y = new int[train.l][];
		
		for(int i = 0; i < tvindex[0].length; i++) {
			train.x[i] = prob.x[tvindex[0][i]];
			train.y[i] = prob.y[tvindex[0][i]];
		}
		
		Problem valid = new Problem();
		valid.bias = prob.bias;
		valid.l = tvindex[0].length;
		valid.n = prob.n;
		valid.x = new DataPoint[valid.l][];
		valid.y = new int[valid.l][];
		
		for(int i = 0; i < tvindex[1].length; i++) {
			valid.x[i] = prob.x[tvindex[1][i]];
			valid.y[i] = prob.y[tvindex[1][i]];
		}
		
		double bestperf = Double.POSITIVE_INFINITY;
		double bestc = cs[0];
		for(int i = 0; i < cs.length; i++) {
			Parameter pa = new Parameter(cs[i], 1000, 0.001);
			double perf = validate(train, valid, pa);
			if(perf < bestperf) {
				bestperf = perf;
				bestc = cs[i];
			}
		}
		
		Parameter pa = new Parameter(bestc, 1000, 0.001);
		DataPoint[][] weight = new DataPoint[this.ulabels.length][];
		for(int i = 0; i < this.ulabels.length; i++) {
			int label = this.ulabels[i];
			int[] bl = getBinaryLabels(prob.y, label);
			double[] loss = new double[1];
			weight[i] = Linear.train(prob, bl, pa, null, loss, null, 0);
		}
		
		return weight;
	}
	
	/**
	 * 
	 */
	public double validate(Problem train, Problem valid, Parameter param) {
		DataPoint[][] weight = new DataPoint[this.ulabels.length][];
		for(int i = 0; i < this.ulabels.length; i++) {
			int label = this.ulabels[i];
			int[] bl = getBinaryLabels(train.y, label);
			double[] loss = new double[1];
			weight[i] = Linear.train(train, bl, param, null, loss, null, 0);
		}
		
		int[][] pl = predict(valid.x, weight);
		double hammingLoss = Measures.averageSymLoss(valid.y, pl);
		return hammingLoss;
	}
	
	/**
	 * 
	 */
	public int[][] predict(DataPoint[][] samples, DataPoint[][] weight) {
		double[][] pv = new double[samples.length][weight.length];
		for(int i = 0; i < weight.length; i++) {
			double[] w = SparseVector.sparseVectorToArray(weight[i], this.prob.n);
			for(int j = 0; j < samples.length; j++) {
				pv[j][i] = SparseVector.innerProduct(w, samples[j]);
			}
		}
		
		int[][] pl = new int[samples.length][];
		for(int i = 0; i < pv.length; i++) {
			double[] tpv = pv[i];
			int counter = 0;
			for(int j = 0; j < tpv.length; j++) {
				if(tpv[j] > 0) {
					counter++;
				}
			}
			
			pl[i] = new int[counter];
			counter = 0;
			for(int j = 0; j < tpv.length; j++) {
				if(tpv[j] > 0) {
					pl[i][counter++] = this.ulabels[j];
				}
			}
		}
		return pl;
	}
	
	/**
	 * 
	 */
	public static int[] getBinaryLabels(int[][] y, int label) {
		int[] bl = new int[y.length];
		for(int i = 0; i < y.length; i++) {
			if(Contain.contain(y[i], label)) {
				bl[i] = 1;
			} else {
				bl[i] = -1;
			}
		}
		return bl;
	}
	
	/**
	 * 
	 */
	public double sigmoid(double x) {
		return 1.0 / (1.0 + Math.exp(-x));
	}
	
	/**
	 * 
	 */
	public double[] object(int[][] y, int label) {
		double[] result = new double[y.length];
		for(int i = 0; i < y.length; i++) {
			if(Contain.contain(y[i], label)) {
				result[i] = 1.0;
			} else {
				result[i] = 0.0;
			}
		}
		return result;
	}
	
	/**
	 * 
	 */
	public double[][] predictValues(DataPoint[][] x) {
		double[][] pv = new double[x.length][this.ulabels.length];
		for(int i = 0; i < this.ulabels.length; i++) {
			double[] w = SparseVector.sparseVectorToArray(this.weight[i], this.prob.n);
			for(int j = 0; j < x.length; j++) {
				pv[j][i] = SparseVector.innerProduct(w, x[j]);
			}
		}
		return pv;
	}
	
	/**
	 * 
	 */
	public void sigmoid(double[][] pv) {
		for(int i = 0; i < pv.length; i++) {
			for(int j =0; j < pv[i].length; j++) {
				pv[i][j] = sigmoid(pv[i][j]);
			}
		}
	}
}
