package com.rssvm;

import java.io.IOException;
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

import com.tools.Statistics;

public class RevisedRSVM {
	private Structure 		structure;
	private DataPoint[][] 	weights;
	private Problem 		prob;
	private Parameter 		param;
	private double 			precision;
	private DataPoint[][] 	all_alpha;
	private int[] 			labels;
	private double[] 		thresholds;

	private DataPoint[][] 	validWeight;
	private double 			B;

	public RevisedRSVM(Structure structure, Problem prob, Parameter param, double precision, double B) throws IOException {
		this.structure 	= structure;
		this.param 		= param;
		this.prob 		= prob;
		this.weights 	= new DataPoint[structure.getTotleVertex()][];
		this.precision 	= precision;
		this.validWeight = new DataPoint[structure.getTotleVertex()][];
		this.labels		= Statistics.getUniqueLabels(this.prob.y);
		this.thresholds	= new double[this.labels.length];
		this.thresholds = new double[labels.length];
		this.B 			= B;
	}
	
	public DataPoint[][] train(Problem train, Problem valid, Parameter param) throws IOException {
		int[] nodes = this.structure.getAllNodes();
		this.all_alpha = new DataPoint[this.structure.getTotleVertex()][];
		double 			obj 		= 0;
		double[] 		loss 		= new double[1];
		double 			totleLoss 	= 0;
		DataPoint[][] 	w = new DataPoint[this.structure.getTotleVertex()][];
		int 		id;
		double 		delta 	= 0;
		double 		lastObj = 0;
		int 		tc 		= 0;
		while(tc < param.getMaxIteration()) {
			totleLoss 	= 0;
			obj 		= 0;			
			int[] rs = RandomSequence.randomSequence(nodes.length);
			for(int i = 0; i < nodes.length; i++) {
				id = nodes[rs[i]];
				if(!structure.isLeaf(id)) {
					updataInnerNode(w, id, this.B);
				} else {
					updateLeafNode(w, id, train, param, loss, this.B);
					totleLoss += loss[0];
				}
			}		
			
			int[][] pre = predict(w, valid.x);
			double micro_f1 = Measures.microf1(this.labels, valid.y, pre);
			double macro_f1 = Measures.macrof1(this.labels, valid.y, pre);
			System.out.println("Micro-F1 = " + micro_f1 + ", Macro-F1 = " + macro_f1);
			
			for(int i = 0; i < nodes.length; i++) {
				id 	= 	nodes[i];
				obj += 	getRegularTerm(w, id);
			}
			
			obj += param.getC() * totleLoss;
			
			delta = Math.abs(obj - lastObj) / lastObj;
			
			if(delta <= this.precision) {
				break;
			}
			lastObj = obj;
			tc++;
		}
		
		this.weights = w;
		return w;
	}
	
	/**
	 * 构造叶节点id的标签
	 * */
	public int[] getLabel(int[][] y, int id) {
		int[] result = new int[y.length];
		int[] temp;
		for(int i = 0; i < result.length; i++) {
			temp = y[i];
			result[i] = -1;
			for(int j = 0; j < temp.length; j++) {
				if(temp[j] == id) {
					result[i] = 1;
					break;
				}
			}
		}
		return result;
	}
	
	/**
	 * 更新中间节点权值
	 * */
	public void updataInnerNode(DataPoint[][] w, int id, double B) {
		int pid = structure.getParent(id);
		int[] childId = structure.getChildren(id);
		
		double totleNieghbour = 0;
		DataPoint[] sum = null;
		if(pid != -1) {
			totleNieghbour++;
			sum = SparseVector.copyScaleVector(w[pid], 1);
		}
		
		if(childId != null && childId.length != 0) {
			for(int i = 0; i < childId.length; i++) {
				totleNieghbour++;
				sum = SparseVector.addVector(sum, w[childId[i]], prob.n);
			}
		}
		w[id] = SparseVector.copyScaleVector(sum, (1 / (totleNieghbour + 2 * B)));
		
	}
	
	
	/**
	 * 
	 * */
	public int[] getLabels(int[] ids, int[][] y) {
		int[] result = new int[y.length];
		boolean flag = false;
		int j;
		for(int i = 0; i < result.length; i++) {
			flag = false;
			for(j = 0; j < ids.length; j++) {
				if(Contain.contain(y[i], ids[j])) {
					flag = true;
					break;
				}
			}
			
			if(flag == true) {
				result[i] = 1;
			} else {
				result[i] = -1;
			}
		}
		
		return result;
	}
	
	/**
	 * 	MNIST 
	 * */
	public void updateLeafNode(DataPoint[][] w, int id, Problem prob, Parameter param, double[] loss, double B) {
		
		
		int[] label = getLabels(id, prob.y);
			
		int nops = numOfPositiveSamples(label);
		
		if(nops == 0) {
			w[id] = null;							//权值为null，导致最后计算w * x 时返回值为0。异常处理
			return;
		}
		int pid = structure.getParent(id);
		DataPoint[] parent = null;
		if(pid != -1) {
			parent = w[pid];
		}

		w[id] = Linear.revisedTrain(prob, label, param, parent, loss, this.all_alpha, id, B);
	}
	
	
	/**
	 * 求节点与父节点权值差的模
	 * */
	public double getRegularTerm(DataPoint[][] w, int id) {
		double result = 0;
		int pid = this.structure.getParent(id);
		DataPoint[] parent = null;
		if(pid != -1) {
			parent = w[pid];
		}
		DataPoint[] wid = w[id];
		DataPoint[] sub = SparseVector.subVector(wid, parent, prob.n);
		if(sub != null && sub.length != 0) {
			result = 0.5 * SparseVector.innerProduct(sub, sub, prob.n);
		} else {
			result = 0;
		}
		return result;
	}

	/**
	 * 预测单个样本类别
	 * */
	public int[] predictSingelSample(DataPoint[][] weight, DataPoint[] sample) {
		int[] leaves = this.labels;
		double predicti = 0;
		double[] predictValues = new double[leaves.length];
		int[] result = null;
		
		for(int i = 0; i < leaves.length; i++) {
			if(weight[leaves[i]] == null) {				//对权值为空的节点不计算输出。问题出在向量相乘时有一个为null返回值为0，而不是抛出异常
				predicti = -Double.MAX_VALUE;
			} else {
				predicti = SparseVector.innerProduct(weight[leaves[i]], sample, prob.n);
			}
			predictValues[i] = predicti;
		}
		
//		result = getPredict(leaves, predictValues, this.thresholds);
		if(result == null || result.length == 0) {
			result = getPredict(leaves, predictValues);
		}
		return result;
	}
	
	/**
	 * 根据叶节点输出值返回预测类标
	 * */
	public int[] getPredict(int[] labels, double[] predictValues) {
		boolean allNegative = true;
		int i;
		for(i = 0; i < predictValues.length; i++) {
			if(predictValues[i] >= 0) {
				allNegative = false;
				break;
			}
		}
		
		
		int[] result = null;
		if(allNegative) {
//System.out.println("all negative");
			int index = -1;
			double max = Double.NEGATIVE_INFINITY;
			
			for(i = 0; i < predictValues.length; i++) {
				if(predictValues[i] > max) {
					max = predictValues[i];
					index = i;
				}
			}
			
			result = new int[1];
			result[0] = labels[index];
			return result;
			
		} else {
			int counter = 0;
			for(i = 0; i < predictValues.length; i++) {
				if(predictValues[i] >= 0) {
					counter++;
				}
			}
			result = new int[counter];
			
			counter = 0;
			for(i = 0; i < predictValues.length; i++) {
				if(predictValues[i] >= 0) {
					result[counter++] = labels[i];
				}
			}
			return result;
		}
	}
	
	
	public int[][] largeScalePredict(DataPoint[][] weight, DataPoint[][] samples) {
		
		int[] labels = this.structure.getLeaves();
		double[][] result = new double[samples.length][labels.length];
		
		double[] w = null;
		
		int i, j;
		
		int id;
		DataPoint[] sample;
		for(i = 0; i < labels.length; i++) {
			id = labels[i];
			w = SparseVector.sparseVectorToArray(weight[id], this.prob.n);
			for(j = 0; j < samples.length; j++) {
				sample = samples[j];
				result[j][i] = SparseVector.innerProduct(w, sample);
			}
		}
		
		int[][] finalLabel = new int[samples.length][1];
		
		int index = -1;
		double max;
		for(i = 0; i < finalLabel.length; i++) {
			
			max = Double.NEGATIVE_INFINITY;
			for(j = 0; j < result[i].length; j++) {
				if(result[i][j] > max) {
					max = result[i][j];
					index= j;
				}
			}
			
			finalLabel[i][0] = labels[index];
		}
		return finalLabel;
	}
	
	
	public double[][] predictValues(DataPoint[][] weight, DataPoint[][] samples) {
		int[] allLabels = this.labels;
		double[][] result = new double[samples.length][allLabels.length];
		
		int i, j;
		double[][] w = new double[weight.length][];
		for(i = 0; i < weight.length; i++) {
			w[i] = SparseVector.sparseVectorToArray(weight[i], this.prob.n);
		}
		
		DataPoint[] sample = null;
		for(i = 0; i < samples.length; i++) {
			sample = samples[i];
			for(j = 0; j < allLabels.length; j++) {
				result[i][j] = SparseVector.innerProduct(w[allLabels[j]], sample);
			}
		}
		return result;
	}
	
	/**
	 *  
	 */
	public int[][] predict(DataPoint[][] weight, DataPoint[][] samples) {
		int[][] result = new int[samples.length][];
		DataPoint[] sample = null;
		double[][] w = new double[weight.length][];
		for(int i = 0; i < w.length; i++) {
			w[i] = SparseVector.sparseVectorToArray(weight[i], this.prob.n);
		}
		
		for(int i = 0; i < samples.length; i++) {
			sample = samples[i];
			result[i] = predictSingleSample(w, sample);
		}
		return result;
	}
	
	/**
	 * 	预测输出，输出为正输出1，输出为负输出-1。
	 * */
	public int[] predictSingleSample(double[][] weight, DataPoint[] sample) {
		int[] las = this.labels;
		double[] preval = new double[las.length];
		int i;
		for(i = 0; i < las.length; i++) {
			preval[i] = SparseVector.innerProduct(weight[las[i]], sample);
		}
		
		int counter = 0;
		for(i = 0; i < preval.length; i++) {
			if(preval[i] > 0) {
				counter++;
			}
		}
		
		int[] result = new int[counter];
		counter = 0;
		for(i = 0; i < preval.length; i++) {
			if(preval[i] > 0) {
				result[counter++] = las[i];
			}
		}
		
		return result;
	}
	
	/**
	 * 
	 * */
	public int[] getLabels(int id, int[][] y) {

		
		int[] result = new int[y.length];
		int[] ty;
		for(int i = 0; i < result.length; i++) {
			ty = y[i];
			if(Contain.contain(ty, id)) {
				result[i] = 1;
			} else {
				result[i] = -1;
			}
			
		}
		return result;
	}
	
	//
	public boolean numInArr(int[] arr, int num) {
		boolean result = false;
		for(int i = 0; i < arr.length; i++) {
			if(num == arr[i]) {
				result = true;
				break;
			}
		}
		return result;
	}

	public Structure getStructure() {
		return structure;
	}

	public void setStructure(Structure structure) {
		this.structure = structure;
	}

	public DataPoint[][] getWeights() {
		return weights;
	}

	public void setWeights(DataPoint[][] weights) {
		this.weights = weights;
	}

	public Problem getProb() {
		return prob;
	}

	public void setProb(Problem prob) {
		this.prob = prob;
	}

	public Parameter getParam() {
		return param;
	}

	public void setParam(Parameter param) {
		this.param = param;
	}

	public double getPrecision() {
		return precision;
	}

	public void setPrecision(double precision) {
		this.precision = precision;
	}
	
	/**
	 * 所有样本标签是否都为-1
	 * */
	public int numOfPositiveSamples(int[] y) {
		int counter = 0;
		for(int i = 0; i < y.length; i++) {
			if(y[i] == 1) {
				counter++;
			}
		}
		return counter;
	}
	
	
	/**
	 * 统计标签中正例的个数
	 * */
	public int getNumOfPositiveSamples(int[] labels) {
		int result = 0;
		if(labels == null) {
			return result;
		}
		for(int i = 0; i < labels.length; i++) {
			if(labels[i] == 1) {
				result++;
			}
		}
		
		return result;
	}
	
	/**
	 * 
	 */
	public double[] predictValues(DataPoint[][] weight, DataPoint[] samples) {
		double[] result = new double[this.labels.length];
		
		DataPoint[] w = null;
		double sum;
		for(int i = 0; i < result.length; i++) {
			w = weight[this.labels[i]];
			if(w == null) {
				result[i] = Double.NEGATIVE_INFINITY;
			} else {
				sum = SparseVector.innerProduct(w, samples, this.prob.n);
				result[i] = sum;
			}
		}
		return result;
	}
	
	public void swap(int[] a, int i, int j) {
		int temp = a[i];
		a[i] = a[j];
		a[j] = temp;
	}
	
	/**
	 * 	
	 * */
	public int[] getLabels(int[][] y, int id) {
		int[] result = new int[y.length];
		for(int i = 0; i < result.length; i++) {
			if(Contain.contain(y[i], id)) {
				result[i] = 1;
			} else {
				result[i] = -1;
			}
		}
		
		return result;
	}


	public DataPoint[][] getValidWeight() {
		return validWeight;
	}

	public void setValidWeight(DataPoint[][] validWeight) {
		this.validWeight = validWeight;
	}
	
	/**
	 * 中间节点也作为类别的预测函数，没有考虑类别一致性问题
	 * */
	public int[][] newPredict(DataPoint[][] w, DataPoint[][] samples) {
		int[] labels = this.structure.levelTraverse();
		int[][] result = new int[samples.length][];
		
		
		double[][] weight = new double[w.length][];
		int i, j;
		for(i = 0; i < weight.length; i++) {
			weight[i] = SparseVector.sparseVectorToArray(w[i], this.prob.n);
		}
		
		DataPoint[] sample = null;
		int counter = 0;
		double[] pv = null;
		for(i = 0; i < samples.length; i++) {
			sample = samples[i];
			pv = new double[labels.length];
			counter = 0;
			for(j = 0; j < labels.length; j++) {
				pv[j] = SparseVector.innerProduct(weight[labels[j]], sample);
				if(pv[j] > 0) {
					counter++;
				}
			}
			
			result[i] = new int[counter];
			counter = 0;
			for(j = 0; j < labels.length; j++) {
				if(pv[j] > 0) {
					result[i][counter++] = labels[j];
				}
			}
		}
		
		return result;
	}
	

	/**
	 * 带有threshold的预测 
	 */
	public int[][] predictWithThreshold(DataPoint[][] xs) {
		int[] labels = this.labels;
		double[][] w = new double[labels.length][];
		for(int i = 0; i < w.length; i++) {
			w[i] = SparseVector.sparseVectorToArray(this.weights[labels[i]], this.prob.n);
		}
		
		int[][] predict = new int[xs.length][];
		double[] pv = new double[labels.length];
		DataPoint[] x = null;
		int counter = 0;
		for(int i = 0; i < xs.length; i++) {
			x = xs[i];
			counter = 0;
			
			for(int j = 0; j < w.length; j++) {
				pv[j] = SparseVector.innerProduct(w[j], x);
				if(pv[j] > this.thresholds[j]) {
					counter++;
				}
			}
			
			predict[i] = new int[counter];
			counter = 0;
			for(int j = 0; j < pv.length; j++) {
				if(pv[j] > this.thresholds[j]) {
					predict[i][counter++] = labels[j];
				}
			}
		}
		return predict;
	}

	public int[] getLabels() {
		return labels;
	}

	public void setLabels(int[] labels) {
		this.labels = labels;
	}
}
