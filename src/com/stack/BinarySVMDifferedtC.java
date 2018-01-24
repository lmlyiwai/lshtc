package com.stack;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Linear;
import com.rssvm.Measures;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;
import com.tools.Contain;
import com.tools.RandomSequence;

public class BinarySVMDifferedtC {
	private Problem prob;
	private Parameter param;
	private int[] ulabels;
	private double[] cs;
	
	public BinarySVMDifferedtC(Problem prob, Parameter param) {
		this.prob = prob;
		this.param = param;
		this.ulabels = getAllLabels(prob.y);
	}
	
	/**
	 *	获得训练集中出现的标签 
	 */
	public int[] getAllLabels(int[][] y) {
		Set<Integer> set = new HashSet<Integer>();
		for(int i = 0; i < y.length; i++) {
			for(int j = 0; j < y[i].length; j++) {
				set.add(y[i][j]);
			}
		}
		
		int[] result = new int[set.size()];
		Iterator<Integer> it = set.iterator();
		for(int i = 0; i < result.length; i++) {
			result[i] = it.next();
		}
		return result;
	}
	
	/**
	 *	交叉验证 ，返回perf
	 */
	public double crossValidation(Problem prob, Parameter param, int n_fold, double c, int labeli) {
		
		param.setC(c);
		
		int label = this.ulabels[labeli];
		int[] labels = getLabels(prob.y, label);
		
		int n = prob.l;
		
		int[] pre = new int[n];
		
		int[] index = RandomSequence.randomSequence(n);
		
		int segLength = n / n_fold;
		
		int vbegin = 0;
		int vend = 0;		
		
		int[] validIndex = null;
		int[] trainIndex = null;
		int counter = 0;
		
		int[] labely = null;
		
		for(int i = 0; i < n_fold; i++) {
			vbegin = i * segLength;
			vend = i * segLength + segLength;
			
			validIndex = new int[vend - vbegin];
			trainIndex = new int[n - validIndex.length];
			
			counter = 0;
			for(int j = vbegin; j < vend; j++) {
				validIndex[counter++] = index[j];
			}
			
			counter = 0;
			for(int j = 0; j < vbegin; j++) {
				trainIndex[counter++] = index[j];
			}
			for(int j = vend; j < n; j++) {
				trainIndex[counter++] = index[j];
			}
			
			Problem train = new Problem();
			train.l = trainIndex.length;
			train.n = prob.n;
			train.bias = prob.bias;
			train.x = new DataPoint[trainIndex.length][];
			train.y = new int[trainIndex.length][1];
			
			labely = new int[train.l];
			
			counter = 0;
			for(int j = 0; j < trainIndex.length; j++) {
				train.x[counter] = prob.x[trainIndex[j]];
				train.y[counter][0] = labels[trainIndex[j]];
				labely[counter] = labels[trainIndex[j]];
				counter++;
			}
			
			Problem valid = new Problem();
			valid.l = validIndex.length;
			valid.n = prob.n;
			valid.bias = prob.bias;
			valid.x = new DataPoint[validIndex.length][];
			valid.y = new int[validIndex.length][1];
			
			counter = 0;
			for(int j = 0; j < validIndex.length; j++) {
				valid.x[counter] = prob.x[validIndex[j]];
				valid.y[counter][0] = labels[validIndex[j]];
				counter++;
			}
			
			double[] tloss = new double[1];
			DataPoint[] w = Linear.train(train, labely, param, null, tloss, null, 0);
			double[] validpv = predictValues(w, valid.x);
			int[] validPre = predictLabels(validpv);
			
			for(int j = 0; j < validIndex.length; j++) {
				pre[validIndex[j]] = validPre[j];
			}
		}
		
		double performance = 0;
//		performance = accuracy(labels, pre);
		performance = f1(labels, pre);
		return performance;
	}
	
	/**
	 * 返回最佳k,c
	 */
	public double binaryCrossValidation(Problem prob, Parameter param, int n_fold, double[] c, int labeli) {
		double[] perf = new double[c.length];
		
		double bestPerf = Double.NEGATIVE_INFINITY;
		int row = 0;
		int label = this.ulabels[labeli];
		for(int i = 0; i < c.length; i++) {
			double tc = c[i];
			param.setC(tc);    
			perf[i] = crossValidation(prob, param, n_fold, tc, label);
			if(perf[i] > bestPerf) {
				bestPerf = perf[i];
				row = i;
			}
		}
		double result = c[row];
		System.out.println("label " + label + ", c = " + result);
		return result;
	}
	
	/**
	 * 
	 */
	public void allCrossValidation(Problem prob, Parameter param, int n_fold, double[] c) {
		double[] cs = new double[this.ulabels.length];
		for(int i = 0; i < this.ulabels.length; i++) {
			cs[i] = binaryCrossValidation(prob, param, n_fold, c, i);
		}
		this.cs = cs;
	}
	
	public DataPoint[][] train(Problem prob, Parameter param) {
		DataPoint[][] w = new DataPoint[this.ulabels.length][];
		int label;
		int[] y;
		double[] tloss = new double[1];
		for(int i = 0; i < this.ulabels.length; i++) {
			label = this.ulabels[i];
			y = getLabels(prob.y, label);
			param.setC(this.cs[i]);
			w[i] = Linear.train(prob, y, param, null, tloss, null, 0);
		}
		return w;
	} 
	
	/**
	 *	 预测输出值
	 */
	public double[][] predictValues(DataPoint[][] w, DataPoint[][] x) {
		int n = this.ulabels.length;
		
		double[][] pv = new double[x.length][n];
		
		double[] weight;
		DataPoint[] tx;
		
		for(int i = 0; i < w.length; i++) {
			weight = SparseVector.sparseVectorToArray(w[i], this.prob.n);
			for(int j = 0; j < x.length; j++) {
				tx = x[j];
				pv[j][i] = SparseVector.innerProduct(weight, tx); 
			}
		}
		return pv;
	}
	
	/**
	 *	预测类别 
	 */
	public int[][] predict(double[][] pv) {
		int counter = 0;
		int[][] result = new int[pv.length][];
		
		for(int i = 0; i < pv.length; i++) {
			counter = 0;
			for(int j = 0; j < pv[i].length; j++) {
				if(pv[i][j] > 0) {
					counter++;
				}
			}
			
			result[i] = new int[counter];
			counter = 0;
			for(int j = 0; j < pv[i].length; j++) {
				if(pv[i][j] > 0) {
					result[i][counter++] = this.ulabels[j];
				}
			}
		}
		return result;
	}
	/**
	 * 
	 */
	public double[] predictValues(DataPoint[] w, DataPoint[][] x) {
		double[] pv = new double[x.length];
		DataPoint[] tx;
		for(int j = 0; j < x.length; j++) {
			tx = x[j];
			pv[j] = SparseVector.innerProduct(w, tx); 
		}
		return pv;
	}
	
	public int[] predictLabels(double[] pv) {
		int[] result = new int[pv.length];
		for(int i = 0; i < result.length; i++) {
			if(pv[i] > 0) {
				result[i] = 1;
			} else {
				result[i] = -1;
			}
		}
		return result;
	}
	
	/**
	 * 
	 */
	public double accuracy(int[] tl, int[] pl) {
		double counter = 0;
		for(int i = 0; i < tl.length; i++) {
			if(tl[i] == pl[i]) {
				counter = counter + 1;
			}
		}
		
		double performance = counter / tl.length;
		return performance;
	}
	
	/**
	 *	获得id指定类标签 
	 */
	public int[] getLabels(int[][] y, int label) {
		if(y == null) {
			return null;
		}
		
		int[] result = new int[y.length];
		for(int i = 0; i < y.length; i++) {
			if(Contain.contain(y[i], label)) {
				result[i] = 1;
			} else {
				result[i] = -1;
			}
		}
		return result;
	}

	public int[] getUlabels() {
		return ulabels;
	}

	public void setUlabels(int[] ulabels) {
		this.ulabels = ulabels;
	}
	
	public double f1(int[] tl, int[] pre) {
		double tp = Measures.truePositive(tl, pre);
		double fp = Measures.falsePositive(tl, pre);
		double fn = Measures.falseNegative(tl, pre);
		
		double perf = (2 * tp) / (2 * tp + fp + fn);
		return perf;
	}
}
