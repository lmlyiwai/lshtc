package com.flatSvm;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Linear;
import com.rssvm.Measures;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;
import com.structure.Structure;
import com.tools.Contain;
import com.tools.Matrix;
import com.tools.ProcessProblem;
import com.tools.RandomSequence;
import com.tools.Sort;

public class SVMKNN {
	private Problem 		prob;
	private Parameter 		param;
	private int[] 			ulabels;
	private DataPoint[][] 	w;
	private Structure       tree;

	
	public SVMKNN(Problem prob, Parameter param) {
		this.prob = prob;
		this.param = param;
		this.ulabels = ProcessProblem.getUniqueLabels(prob.y);
	}
	/*
	 *  每训练一个权将其写入文件
	 * @throws IOException 
	 */
	public DataPoint[][] trainWithInnerNode(Problem prob, Parameter param) throws IOException {
		int[] nodes = this.ulabels;
		
		DataPoint[][] weights = new DataPoint[nodes.length][];
		
		for(int i = 0; i < nodes.length; i++) {
			int label = nodes[i];
			int[] labels = getBinaryLabels(prob.y, label);
			boolean allNegative = isAllNegative(labels);
			//样本中不包含正例
			if(allNegative) {
				weights[i] = null;
				continue;
			}
			double[] tloss = new double[1];
//			System.out.print(i + ", " + label + ", ");
			long start = System.currentTimeMillis();
			weights[i] = Linear.train(prob, labels, param, null, tloss, null, 0);
			long end = System.currentTimeMillis();
//			System.out.println((end - start) + "ms");
		}

		return weights;
	}
	

	public int[] getBinaryLabels(int[][] y, int label) {
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
	
	/**
	 * 样本经SVM输出值 
	 */
	public double[][] transformSamples(DataPoint[][] x, DataPoint[][] w, int dim) {
		if(x == null || w == null) {
			return null;
		}
		
		double[][] fullw = new double[w.length][];
		for(int i = 0; i < w.length; i++) {
			fullw[i] = SparseVector.sparseVectorToArray(w[i], dim);
		}
		
		double[][] result = new double[x.length][w.length];
		for(int i = 0; i < x.length; i++) {
			for(int j = 0; j < fullw.length; j++) {
				result[i][j] = SparseVector.innerProduct(fullw[j], x[i]);
			}
		}
		return result;
	}
	
	public boolean isAllNegative(int[] y) {
		boolean flag = true;
		for(int i = 0; i< y.length; i++) {
			if(y[i] > 0) {
				flag = false;
				break;
			}
		}
		return flag;
	}
	
	/**
	 * 预测样本类标 
	 */
	public int[][] predictKnearLabels(double[][] trainSample, int[][] y, double[][] testSample, int k) {
		if(trainSample == null || testSample == null) {
			return null;
		}
		
		int[] nodes = this.ulabels;
		int[][] ys = new int[nodes.length][];
		for(int i = 0; i < ys.length; i++) {
			ys[i] = getBinaryLabels(y, nodes[i]);
		}
		
		int[][] pl = new int[testSample.length][];
		
		for(int i = 0; i < testSample.length; i++) {
			System.out.println("predict " + i);
			double[] tx = testSample[i];
			int[] tpl = new int[nodes.length];
			
			long start = System.currentTimeMillis();
			for(int j = 0; j < nodes.length; j++) {
				int[] path = this.tree.getPathToRoot(nodes[j]);
				int[] index = getIndex(nodes, path);
				tpl[j] = getLabel(trainSample, tx, ys[j], index, k);
			}
			
			int count = 0;
			for(int j = 0; j < tpl.length; j++) {
				if(tpl[j] > 0) {
					count++;
				}
			}
			
			int[] ntpl = new int[count];
			count = 0;
			for(int j = 0; j < nodes.length; j++) {
				if(tpl[j] > 0) {
					ntpl[count++] = nodes[j];
				}
			}
			
			long end = System.currentTimeMillis();
			
			System.out.println(i + ", time = " + (end - start));
			pl[i] = ntpl;
		}
		return pl;
	} 
	
	public int getLabel(double[][] trainSample, double[] tx, int[] y, int[] index, int k) {
		double[] dis = new double[trainSample.length];
		for(int i = 0; i < trainSample.length; i++) {
			double[] sub = new double[index.length];
			for(int j = 0; j < sub.length; j++) {
				sub[j] = trainSample[i][index[j]] - tx[index[j]];
			}
			dis[i] = SparseVector.innerProduct(sub, sub);
		}
		
		int[] sortIndex = Sort.getIndexBeforeSort(dis);
		int result = 0;
		for(int i = 0; i < k; i++) {
			result += y[sortIndex[i]];
		}
		
		if(result > 0) {
			return 1;
		} else {
			return -1;
		}
	}
	
	public int[] getIndex(int[] nodes, int[] path) {
		int[] index = new int[path.length - 1];
		int count = 0;
		for(int i = 0; i < path.length; i++) {
			for(int j = 0; j < nodes.length; j++) {
				if(path[i] == nodes[j]) {
					index[count++] = j;
					break;
				}
			}
		}
		return index;
	}
	
	
	public double[] crossValidation(Problem prob, Parameter param, int n_fold, int k) throws IOException {
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
			
			DataPoint[][] w = trainWithInnerNode(train, param);
			
			double[][] trainSample = transformSamples(train.x, w, train.n);
			scale(trainSample);
			
			double[][] validSample = transformSamples(valid.x, w, valid.n);
			scale(validSample);
			
			long start = System.currentTimeMillis();
			int[][] predictLabel = predictKnearLabels(trainSample, train.y, validSample, k);
			long end = System.currentTimeMillis();
			System.out.println(i + ", fold, time = " + (end - start));
			for(int j = 0; j < validIndex.length; j++) {
				pre[validIndex[j]] = predictLabel[j];
			}
		}
		double microf1 = Measures.microf1(this.ulabels, prob.y, pre);
		double macrof1 = Measures.macrof1(this.ulabels, prob.y, pre);
		double hammingloss = Measures.averageSymLoss(prob.y, pre);
		System.out.println(", Micro-F1 = " + microf1 + ", Macro-F1 = " 
				+ macrof1 + ", Hamming Loss = " + hammingloss);
		double[] perf = {microf1, macrof1, hammingloss};
		return perf;
	}
	
	/**
	 *	向量归一化 
	 */
	public void scale(double[][] pv) {
		double norm = 0;
		double[] temp = null;
		for(int i = 0; i < pv.length; i++) {
			temp = pv[i];
			norm = SparseVector.innerProduct(temp, temp);
			norm = Math.pow(norm, 0.5);
			for(int j = 0; j < pv[i].length; j++) {
				pv[i][j] = pv[i][j] / norm;
			}
		}
	}
	public DataPoint[][] getW() {
		return w;
	}
	public void setW(DataPoint[][] w) {
		this.w = w;
	}
	public Structure getTree() {
		return tree;
	}
	public void setTree(Structure tree) {
		this.tree = tree;
	}
	public int[] getUlabels() {
		return ulabels;
	}
	public void setUlabels(int[] ulabels) {
		this.ulabels = ulabels;
	}
	
	
}
