package com.rssvm;

import java.util.ArrayList;
import java.util.List;

import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;
import com.structure.Structure;
import com.tools.Contain;

public class RevisedSVM {
	private Structure 		structure;
	private Problem 		prob;
	private Parameter 		param;
	private DataPoint[][] 	w;
	
	public RevisedSVM(Structure structure, Problem prob, Parameter param) {
		this.structure = structure;
		this.prob = prob;
		this.param = param;
		this.w = new DataPoint[this.structure.getTotleVertex()][];
	}
	
	/**
	 * 
	 */
	public void train() {
		int[] ids = this.structure.levelTraverse();
		
		int id;
		int i;
		int[] y;
		Problem pro;
		double nd;
		for(i = 0; i < ids.length; i++) {
			id = ids[i];
			System.out.println(id);
			y = this.getLabels(id);
			pro = this.getProbs(id);
			nd = this.structure.getDes(id).length;
			this.w[id] = Linear.train(pro, y, this.param, nd);
		}
	}
	
	/**
	 * 
	 */
	public int[][] predict(DataPoint[][] samples) {
		int[][] result = new int[samples.length][];
		double[][] weight = new double[this.w.length][];
		
		int i;
		for(i = 0; i < weight.length; i++) {
			weight[i] = SparseVector.sparseVectorToArray(this.w[i], this.prob.n);
		}
		
		DataPoint[] sample;
		for(i = 0; i < samples.length; i++) {
			sample = samples[i];
			result[i] = predictSingleSample(weight, sample);
		}
		return result;
	}
	
	/**
	 * 
	 */
	public int[] predictSingleSample(double[][] weigth, DataPoint[] sample) {
		int[] labels = this.structure.levelTraverse();
		
		int[] path = null;
		int i, j;
		int id;
		int root = this.structure.getRoot();
		double pv = 0;
		List<Integer> list = new ArrayList<Integer>();
		
		for(i = 0; i < labels.length; i++) {
			id = labels[i];
			path = this.structure.getPathToRoot(id);
			pv = 0;
			for(j = 0; j < path.length; j++) {
				if(path[j] != root) {
					pv += SparseVector.innerProduct(weigth[path[j]], sample);
				}
			}
			
			if(pv > 0) {
				list.add(id);
			}
		}
		
		int[] result = new int[list.size()];
		for(i = 0; i < result.length; i++) {
			result[i] = list.get(i);
		}
		return result;
	}
	
	/**
	 *	获得类标 
	 */
	public int[] getLabels(int id) {
		int[] des = this.structure.getDes(id);
		int[] result = new int[des.length * this.prob.l];
		
		int i, j;
		int n = this.prob.l;
		int tid;
		for(i = 0; i < des.length; i++) {
			tid = des[i];
			for(j = 0; j < this.prob.l; j++) {
				if(Contain.contain(this.prob.y[j], tid)) {
					result[i * n + j] = 1;
				} else {
					result[i * n + j] = -1;
				}
			}
		}
		return result;
	}
	
	/**
	 * 只是将训练样本重复多次，重复次数等于以id为根的子树节点个数
	 */
	public Problem getProbs(int id) {
		int[] des = this.structure.getDes(id);
		int n = des.length;
		
		Problem newprob = new Problem();
		newprob.l       = this.prob.l * n;
		newprob.n       = this.prob.n;
		newprob.bias    = this.prob.bias;
		newprob.x       = new DataPoint[newprob.l][];
		newprob.y       = new int[newprob.l][];
		
		int m = this.prob.l;
		
		int i, j;
		for(i = 0; i < n; i++) {
			for(j = 0; j < m; j++) {
				newprob.x[i * m + j] = this.prob.x[j];
				newprob.y[i * m + j] = this.prob.y[j];
			}
		}
		return newprob;
	}
	
	/**
	 *	返回正例数量不为0的类别 
	 */
	public int[] getValidateLabels(int[][] labels) {
		int[] allLabels = this.structure.levelTraverse();
		
		List<Integer> list = new ArrayList<Integer>();
		int i, j;
		int id;
		int counter = 0;
		for(i = 0; i < allLabels.length; i++) {
			id = allLabels[i];
			counter = 0;
			for(j = 0; j < labels.length; j++) {
				if(Contain.contain(labels[j], id)) {
					counter++;
				}
			}
			
			if(counter != 0) {
				list.add(id);
			}
		}
		
		int[] result = new int[list.size()];
		for(i = 0; i < result.length; i++) {
			result[i] = list.get(i);
		}
		
		return result;
	}
}
