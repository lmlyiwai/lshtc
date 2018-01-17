package com.consistance;

import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Linear;
import com.rssvm.Measures;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;
import com.structure.Structure;
import com.tools.Contain;
import com.tools.RandomSequence;

public class ConsistentSVM {
	private Problem 		prob;
	private Parameter 		param;
	private Structure   	tree;
	private DataPoint[][] 	w;
	private int[] 			labels;
	
	public ConsistentSVM(Problem prob, Parameter param, Structure tree) {
		this.prob = prob;
		this.param = param;
		this.tree = tree;
		this.labels = tree.levelTraverse();
	}
	
	public void train(Problem prob, Parameter param) {
		int[] travel = this.tree.levelTraverse();
		DataPoint[][] tw = new DataPoint[travel.length][];
		int root = this.tree.getRoot();
		
		int[] ry = new int[prob.l];
		for(int i = 0; i < ry.length; i++) {
			ry[i] = 1;
		}
		double[] loss = new double[1];
		DataPoint[] rootw = Linear.train(prob, ry, param, null, loss, null, 0);
		
		for(int i = 0; i < travel.length; i++) {
			int id = travel[i];
			int[] y = getLabels(prob.y, id);
			if(this.tree.getParent(id) == root) {
				tw[i] = Linear.newtrain(prob, y, param, rootw);
			} else {
				int pid = this.tree.getParent(id);
				int index = getIndex(travel, pid);
				DataPoint[] parent = tw[index];
				tw[i] = Linear.newtrain(prob, y, param, parent);
			}
		}
		this.w = tw;
	}
	
	public int[] getLabels(int[][] y, int label) {
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
	
	public int getIndex(int[] y, int toFind) {
		int index = -1;
		for(int i = 0; i < y.length; i++) {
			if(y[i] == toFind) {
				index = i;
				break;
			}
		}
		return index;
	}
	
	/**
	 * 
	 */
	public int[] predictSingleLabels(DataPoint[] x) {
		double[] pv = new double[this.w.length];
		int counter = 0;
		for(int i = 0; i < pv.length; i++) {
			pv[i] = SparseVector.innerProduct(w[i], x);
			if(pv[i] > 0) {
				counter = counter + 1;
			}
		}
		
		int[] pl = new int[counter];
		counter = 0;
		for(int i = 0; i < pv.length; i++) {
			if(pv[i] > 0) {
				pl[counter] = this.labels[i];
				counter = counter + 1;
			}
		}
		return pl;
	}
	
	/**
	 * 
	 */
	public int[][] predict(DataPoint[][] xs) {
		int[][] pl = new int[xs.length][];
		for(int i = 0; i < pl.length; i++) {
			pl[i] = predictSingleLabels(xs[i]);
		}
		return pl;
	}

	public DataPoint[][] getW() {
		return w;
	}

	public void setW(DataPoint[][] w) {
		this.w = w;
	}

	public int[] getLabels() {
		return labels;
	}

	public void setLabels(int[] labels) {
		this.labels = labels;
	}
	
	/**
	 *	交叉验证 ，返回性能
	 */
	public double[] crossValidation(Problem prob, Parameter param, int n_fold) {
		int n = prob.l;
		
		int[][] pre = new int[n][];
		
		int[] index = RandomSequence.randomSequence(n);
		
		int segLength = n / n_fold;
		
		int vbegin = 0;
		int vend = 0;		
		
		int[] validIndex = null;
		int[] trainIndex = null;
		int counter = 0;
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
			train.y = new int[trainIndex.length][];
			
			counter = 0;
			for(int j = 0; j < trainIndex.length; j++) {
				train.x[counter] = prob.x[trainIndex[j]];
				train.y[counter] = prob.y[trainIndex[j]];
				counter++;
			}
			
			Problem valid = new Problem();
			valid.l = validIndex.length;
			valid.n = prob.n;
			valid.bias = prob.bias;
			valid.x = new DataPoint[validIndex.length][];
			valid.y = new int[validIndex.length][];
			
			counter = 0;
			for(int j = 0; j < validIndex.length; j++) {
				valid.x[counter] = prob.x[validIndex[j]];
				valid.y[counter] = prob.y[validIndex[j]];
				counter++;
			}
			
			train(train, param);
			
			int[][] predictLabel = predict(valid.x);
			
			for(int j = 0; j < validIndex.length; j++) {
				pre[validIndex[j]] = predictLabel[j];
			}
		}
		double microf1 = Measures.microf1(this.labels, prob.y, pre);
		double macrof1 = Measures.macrof1(this.labels, prob.y, pre);
		double hammingloss = Measures.averageSymLoss(prob.y, pre);
		System.out.println("c = " + param.getC() + ", c1 = " + param.getC1() + ", Micro-F1 = " + microf1 + ", Macro-F1 = " 
				+ macrof1 + ", Hamming Loss = " + hammingloss);
		
		double[] perf = {microf1, macrof1, hammingloss};
		return perf;
	}
}
