package com.rssvm;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;
import com.structure.Structure;
import com.tools.Contain;
import com.tools.RandomSequence;

public class EqualUnitedSVM {
	private Structure 		structure;
	private Parameter 		param;
	private Problem 		prob;
	private double[][]		alphas;
	private double[][] 		weights;
	private int[]			labels;
	private int 			maxIteration;
	private double 			precision;
	
	private static  Random random = new Random();
	
	public EqualUnitedSVM(Structure structure, Parameter param, Problem prob, int maxIteration, double precision) {
		this.structure = structure;
		this.param = param;
		this.prob = prob;
		this.maxIteration = maxIteration;
		this.weights = new double[this.structure.getAllNodes().length][this.prob.n];
		this.labels = this.structure.levelTraverse();
		this.precision = precision;
	}
	
	/**
	 *	train 
	 */
	public void train(Problem prob, Parameter param) {
		int[] nodes = this.structure.levelTraverse();
		int counter = 0;
		int[] rs;
		double[][] alphas = new double[this.structure.getAllNodes().length][prob.l];
		this.alphas = alphas;
		
		double primalObj = Double.POSITIVE_INFINITY;
		double dualObj = Double.NEGATIVE_INFINITY;
		
		while(true) {
			System.out.print(counter);
			rs = RandomSequence.randomSequence(nodes.length);
			int id;
			for(int i = 0; i < nodes.length; i++) {
				id = nodes[rs[i]];
//				id = nodes[i];
				updateNode(prob, param, id);
			}
			
			for(int i = 0; i < nodes.length; i++) {
				this.weights[nodes[i]] = getNodeWeight(nodes[i], prob);
			}
			
			primalObj = getPrimalObj(nodes, prob, this.param.getC());
			dualObj = getDualObj(nodes, prob, this.alphas);
					
			System.out.print(", Primal Object = " + primalObj);
			System.out.print(", Dual Object = " + dualObj);
			System.out.println();
			
			counter++;
			
			if(counter >= this.maxIteration || (primalObj - dualObj) < this.precision) {
				break;
			}
		}
		
	}
	
	/**
	 * 获得关于id的两类分类标签，包含id为1，不包含id为-1.
	 */
	public int[] getBinaryLabel(int[][] y, int id) {
		if(y == null) {
			return null;
		}
		
		int[] result = new int[y.length];
		for(int i = 0; i < y.length; i++) {
			if(Contain.contain(y[i], id)) {
				result[i] = 1;
			} else {
				result[i] = -1;
			}
		}
		return result;
	}
	
	/**
	 *	获得以id为根的子树的节点权值 
	 */
	public double[] getNodeWeight(int id, Problem prob) {
		int[] des = this.structure.getDes(id);
		double[] weight = new double[prob.n];
		
		int i, j;
		int[] y = null;
		double[] alpha = null;
		int tid;
		for(i = 0; i < des.length; i++) {
			tid = des[i];
			y = getBinaryLabel(prob.y, tid);
			alpha = this.alphas[tid];
			
			for(j = 0; j < prob.l; j++) {
				for(DataPoint dp : prob.x[j]) {
					weight[dp.index - 1] += alpha[j] * y[j] * dp.value;
				}
			}
		}
		
		double scale = 1 / (2 * (double)1);
		for(i = 0; i < weight.length; i++) {
			weight[i] *= scale;
		}
		return weight;
	}
	
	/**
	 *	更新指定节点id 
	 */
	public void updateNode(Problem prob, Parameter param, int id) {
		int[] path = this.structure.getPathToRoot(id);
		int root = this.structure.getRoot();
		int[] nodes = filterArray(path, root);
		int[] y;
		double[][] w = new double[nodes.length][];
		double[] lambda = new double[nodes.length];
		
		y = getBinaryLabel(prob.y, id);
		
		for(int i = 0; i < nodes.length; i++) {
			w[i] = getNodeWeight(nodes[i], prob);
			lambda[i] = 1;
		}
		
		updateAlpha(prob, param, y, w, lambda, id);
	}

	
	/**
	 *	取节点id所对应的alpha优化 
	 */
	public void updateAlpha(Problem prob, Parameter param, int[] y, double[][] w, double[] lambda, int id) {
		int l 				= prob.l;
		int[] index 		= new int[l];
		double[] alpha 		= new double[l];
		int active_size 	= l;
		double[] QD			= new double[l];
		int i, j, s, iter = 0;
		double C, d, G;
		
		double PG;
		double PGmax_old = Double.POSITIVE_INFINITY;
		double PGmin_old = Double.NEGATIVE_INFINITY;
		double PGmax_new, PGmin_new;
		
		for(i = 0; i < l; i++) {
			alpha[i] = this.alphas[id][i];
		}
		
		for(i = 0; i < l; i++) {
			QD[i] = 0;
			double temp = 0;
			for(DataPoint dp : prob.x[i]) {
				double val = dp.value;
				temp += val * val;
			}
			
			for(j = 0; j < lambda.length; j++) {
				QD[i] += temp / (2 * lambda[j]);
			}
			
			index[i] = i;
		}
		
		while(iter < param.getMaxIteration()) {
			PGmax_new = Double.NEGATIVE_INFINITY;
			PGmin_new = Double.POSITIVE_INFINITY;
			
			for(i = 0; i < active_size; i++) {
				int k = i + random.nextInt(active_size - i);
				swap(index, i, k);
			}
			
			for(s = 0; s < active_size; s++) {
				i = index[s];
				G = 0;
				int yi = y[i];
				
				G = getG(yi, w, prob.x[i]);
				C = param.getC();
				
				PG = 0;
				if(alpha[i] == 0) {
					if(G > PGmax_old) {
						active_size--;
						swap(index, s, active_size);
						s--;
						continue;
					} else if(G < 0) {
						PG = G;
					}
				} else if(alpha[i] == C) {
					if(G < PGmin_old) {
						active_size--;
						swap(index, s, active_size);
						s--;
						continue;
					} else if(G > 0) {
						PG = G;
					}
				} else {
					PG = G;
				}
				
				PGmax_new = Math.max(PGmax_new, PG);
				PGmin_new = Math.min(PGmin_new, PG);
				
				if(Math.abs(PG) > 1.0e-12) {
					double alpha_old = alpha[i];
					alpha[i] = Math.min(Math.max(alpha[i] - (G / QD[i]), 0.0), C);
					d = (alpha[i] - alpha_old);
					updateWeight(w, yi, prob.x[i], lambda, d);
				}
			}
			
			iter++;
			if(PGmax_new - PGmin_new <= param.getEps()) {
				if(active_size == l) {
					break;
				} else {
					active_size = l;
					PGmax_old = Double.POSITIVE_INFINITY;
					PGmin_old = Double.NEGATIVE_INFINITY;
					continue;
				}
			}
			
			PGmax_old = PGmax_new;
			PGmin_old = PGmin_new;
			if(PGmax_old <= 0) PGmax_old = Double.POSITIVE_INFINITY;
			if(PGmin_old >= 0) PGmin_old = Double.NEGATIVE_INFINITY;
		}

		for(i = 0; i < l; i++) {
			this.alphas[id][i] = alpha[i];
		}
	}
	
	/**
	 *	过滤数组中特定值 
	 */
	public int[] filterArray(int[] a, int num) {
		List<Integer> list = new ArrayList<Integer>();
		int i;
		for(i = 0; i < a.length; i++) {
			if(a[i] != num) {
				list.add(a[i]);
			}
		}
		
		int[] result = new int[list.size()];
		for(i = 0; i < result.length; i++) {
			result[i] = list.get(i);
		}
		return result;
	}

	/**
	 * 
	 */
	public void swap(int[] a, int i, int j) {
		int temp = a[i];
		a[i] = a[j];
		a[j] = temp;
	}

	
	/**
	 *	计算梯度 
	 */
	public double getG(int y, double[][] w, DataPoint[] x) {
		if(w == null || x == null) {
			return Double.NaN;
		}
		
		double G = 0;
		for(int i = 0; i < w.length; i++) {
			G += y * SparseVector.innerProduct(w[i], x);
		}
		
		G = G - 1;
		return G;
	}

	/**
	 *	更新权值 
	 */
	public void updateWeight(double[][] w, int y, DataPoint[] x, double[] lambda, double d) {
		if(w == null || x == null) {
			return;
		}
		
		double scale = 0;
		for(int i = 0; i < w.length; i++) {
			scale = d * y / (2 * lambda[i]);
			for(DataPoint dp : x) {
				w[i][dp.index - 1] += scale * dp.value;
			}
		}
	}
	
	/**
	 * 预测输出
	 */
	public int[][] predict(DataPoint[][] samples) {
		int[][] result = new int[samples.length][];
		for(int i = 0; i < result.length; i++) {
			result[i] = predictSingleSamples(samples[i]);
		}
		return result;
	}
	
	/**
	 *	预测单个样本 
	 */
	public int[] predictSingleSamples(DataPoint[] sample) {
		int[] labels = this.labels;
		int root = this.structure.getRoot();
		
		List<Integer> list = new ArrayList<Integer>();
		int[] path;
		int id;
		for(int i = 0; i < labels.length; i++) {
			id = labels[i];
			path = this.structure.getPathToRoot(id);
			double pv = 0;
			
			if(path != null) {
				for(int j = 0; j < path.length; j++) {
					if(path[j] != root) {
						pv += SparseVector.innerProduct(this.weights[path[j]], sample);
					}
				}
			}
			
			if(pv > 0) {
				list.add(labels[i]);
			}
		}
		
		int[] result = new int[list.size()];
		for(int i = 0; i < result.length; i++) {
			result[i] = list.get(i);
		}
		return result;
	}
	
	/**
	 *	计算对偶平方项 
	 */
	public double getNodeNormal(int id, Problem prob) {
		int[] des = this.structure.getDes(id);
		double[] weight = new double[prob.n];
		
		int i, j;
		int[] y = null;
		double[] alpha = null;
		int tid;
		for(i = 0; i < des.length; i++) {
			tid = des[i];
			y = getBinaryLabel(prob.y, tid);
			alpha = this.alphas[tid];
			
			for(j = 0; j < prob.l; j++) {
				for(DataPoint dp : prob.x[j]) {
					weight[dp.index - 1] += alpha[j] * y[j] * dp.value;
				}
			}
		}
		
		double scale = 1 / (4 * (double)1);
		double result = scale * SparseVector.innerProduct(weight, weight);
		
		return result;
	}
	
	/**
	 *	节点id对应的alpha值求和 
	 */
	public double getNodeAlphaSum(int id, double[][] alpha) {
		if(alpha == null || id >= alpha.length) {
			return Double.NaN;
		}
		
		double sum = 0;
		for(int i = 0; i < alpha[id].length; i++) {
			sum += alpha[id][i];
		}
		return sum;
	}
	
	/**
	 * 计算对偶函数目标值 
	 */
	public double getDualObj(int[] nodes, Problem prob, double[][] alpha) {
		double item1 = 0;
		double item2 = 0;
		
		int id;
		for(int i = 0; i < nodes.length; i++) {
			id = nodes[i];
			item1 += getNodeNormal(id, prob);
			item2 += getNodeAlphaSum(id, alpha);
		}
		
		return (item2 - item1);
	}
	
	/**
	 * 计算主问题中损失 
	 */
	public double getNodeLoss(int id, Problem prob) {
		int[] path = this.structure.getPathToRoot(id);
		int root = this.structure.getRoot();
		int[] nodes = filterArray(path, root);
		
		double[] w = new double[prob.n];
		
		for(int i = 0; i < nodes.length; i++) {
			w = SparseVector.addVector(w, this.weights[nodes[i]]);
		}
		
		int[] y = getBinaryLabel(prob.y, id);
		
		double loss = 0;
		double kc = 0;
		
		for(int i = 0; i < prob.l; i++) {
			kc = 1 - y[i] * SparseVector.innerProduct(w, prob.x[i]);
			
			kc = Math.max(kc, 0);
			
			loss += kc;
		}
		return loss;
	}
	
	/**
	 *	主问题目标值 
	 */
	public double getPrimalObj(int[] nodes, Problem prob, double c) {
		double item1 = 0;
		double item2 = 0;
		
		int i;
		int id;
		double lambda;
		for(i = 0; i < nodes.length; i++) {
			id = nodes[i];
			lambda = 1;
			
			item1 += lambda * SparseVector.innerProduct(this.weights[id], this.weights[id]);
			
			item2 += getNodeLoss(id, prob);
		}
		
		return (item1 + c * item2);
	}
	
	/**
	 *	获得训练集中正例不为0的类别 
	 */
	public int[] getValidateLabels(int[][] y) {
		int[] labels = this.labels;
		List<Integer> list = new ArrayList<Integer>();
		
		int id;
		int[] yi;
		int counter = 0;
		for(int i = 0; i < labels.length; i++) {
			id = labels[i];
			yi = getBinaryLabel(y, id);
			counter = getNumOfPositive(yi);
			if(counter > 0) {
				list.add(labels[i]);
			}
		}
		
		int[] result = new int[list.size()];
		for(int i = 0; i < result.length; i++) {
			result[i] = list.get(i);
		}	
		return result;
	}
	
	/**
	 *	标签中正例的数目 
	 */
	public int getNumOfPositive(int[] y) {
		if(y == null) {
			return 0;
		}
		
		int counter = 0;
		for(int i = 0; i < y.length; i++) {
			if(y[i] == 1) {
				counter++;
			}
		}
		return counter;
	}
}
