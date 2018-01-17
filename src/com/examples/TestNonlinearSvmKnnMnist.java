package com.examples;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.rssvm.Measures;
import com.sparseVector.SparseVector;
import com.tools.RandomSequence;
import com.tools.Sort;



public class TestNonlinearSvmKnnMnist {

	
	public void gridSerach(svm_problem prob, svm_parameter param, int n_fold, double c, double gamma, int[] k) {
		int[] labels = getUniqueLabels(prob.y);
		
		int n = prob.l;
		
		int[][][] pre = new int[k.length][n][];
		
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
			
			svm_problem train = new svm_problem();
			train.l = trainIndex.length;
			train.x = new svm_node[train.l][];
			train.y = new double[train.l];
			
			counter = 0;
			for(int j = 0; j < trainIndex.length; j++) {
				train.x[counter] = prob.x[trainIndex[j]];
				train.y[counter] = prob.y[trainIndex[j]];
				counter++;
			}
			
			svm_problem valid = new svm_problem(); 
			valid.l = validIndex.length;
			valid.x = new svm_node[valid.l][];
			valid.y = new double[valid.l];
			
			counter = 0;
			for(int j = 0; j < validIndex.length; j++) {
				valid.x[counter] = prob.x[validIndex[j]];
				valid.y[counter] = prob.y[validIndex[j]];
				counter++;
			}
			
			param.C = c;
			param.gamma = gamma;
			
			
			svm_model[] models = train(train, param, labels);
			
			double[][] trainpv = predict_values(train, models);
			scale(trainpv);
			
			double[][] validpv = predict_values(valid, models);
			scale(validpv);
			
			int[][][] temppre = new int[k.length][valid.l][];
			for(int h = 0; h < validpv.length; h++) {
				double[] dis = distance(trainpv, validpv[h]);
				int[] ind = Sort.getIndexBeforeSort(dis);
				for(int m = 0; m < k.length; m++) {
					int[][] ty = getFirstKY(ind, train.y, k[m]);
					int[] tpy = voteLabel(ty);
					temppre[m][h] = tpy;
				}
			}
			
			for(int h = 0; h < valid.l; h++) {
				for(int m = 0; m < k.length; m++) {
					pre[m][validIndex[h]] = temppre[m][h];
				}
			}
		}
		
		int[][] y = doubleArrayToIntMat(prob.y);
		
		for(int i = 0; i < k.length; i++) {		
			double hammingLoss = Measures.averageSymLoss(y, pre[i]);
			double zeroneloss = Measures.zeroOneLoss(y, pre[i]);
			System.out.println("gamma = " + gamma + ", C = " + c + ", K = " + k[i] + ", Hamming Loss = " + hammingLoss + 
					", zero one loss = " + zeroneloss);
		}
	}
	
	/**
	 * 
	 */
	public svm_model[] train(svm_problem prob, svm_parameter param, int[] labels) {
		svm_model[] models = new svm_model[labels.length];
		double[] y = new double[prob.l];
		for(int i = 0; i < y.length; i++) {
			y[i] = prob.y[i];
		}
		
		double label;
		double[] cy;
		for(int i = 0; i < labels.length; i++) {
			label = labels[i];
			cy = getBinaryLabels(y, label);
			prob.y = cy;
			models[i] = svm.svm_train(prob, param);
		}
		
		prob.y = y;
		
		return models;
	} 
	
	/**
	 * 
	 */
	public double[][] predict_values(svm_problem prob, svm_model[] models) {
		double[][] pv = new double[prob.l][models.length];
		svm_model tempModel = null;
		
		double[] predictValue = new double[1];
		for(int i = 0; i < models.length; i++) {
			tempModel = models[i];
			for(int j = 0; j < prob.l; j++) {
				svm.svm_predict_values(tempModel, prob.x[j], predictValue);
				pv[j][i] = predictValue[0];
			}
		}
		return pv;
	}
	
	/**
	 *	预测类别 
	 */
	public int[][] predict(double[][] pv, int[] uniqueLabels) {
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
					result[i][counter++] = uniqueLabels[j];
				}
			}
		}
		return result;
	}
	
	/**
	 * 
	 */
	public int[][] predictMax(double[][] pv, int[] uniqueLabels) {
		int[][] result = new int[pv.length][];
		
		for(int i = 0; i < pv.length; i++) {
			
			result[i] = new int[1];
			int ind = 0;
			double max = Double.NEGATIVE_INFINITY;
			for(int j = 0; j < pv[i].length; j++) {
				if(pv[i][j] > max) {
					max = pv[i][j];
					ind = j;
				}
			}
			result[i][0] = uniqueLabels[ind];
		}
		return result;
	}
	
	/**
	 * 
	 */
	public double[] getBinaryLabels(double[] y, double n) {
		double[] result = new double[y.length];
		for(int i = 0; i < result.length; i++) {
			if(y[i] == n) {
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
	public double[] getBinaryLabels(int[] y, double n) {
		double[] result = new double[y.length];
		for(int i = 0; i < result.length; i++) {
			if(y[i] == n) {
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
	public double[] distance(double[][] pv, double[] p) {
		double[] result = new double[pv.length];
		double[] sub = null;
		for(int i = 0; i < result.length; i++) {
			sub = SparseVector.subVector(pv[i], p);
			result[i] = SparseVector.innerProduct(sub, sub);
		}
		return result;
	}
	
	/**
	 *	 
	 */
	public static int[] getUniqueLabels(double[] y) {
		Set<Integer> set = new HashSet<Integer>();
		for(int i = 0; i < y.length; i++) {
			set.add((int)y[i]);
		}
		int[] result = new int[set.size()];
		Iterator<Integer> it = set.iterator();
		int counter = 0;
		while(it.hasNext()) {
			result[counter++] = it.next();
		}
		return result;
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
	
	/**
	 * 
	 */
	public int[][] getFirstKY(int[] index, int[][] y, int k) {
		int[][] result = new int[k][];
		for(int i = 0; i < k; i++) {
			result[i] = y[index[i]];
		}
		return result;
	}

	
	/**
	 * 
	 */
	public int[][] getFirstKY(int[] index, double[] y, int k) {
		int[][] result = new int[k][1];
		for(int i = 0; i < k; i++) {
			result[i][0] = (int)y[index[i]];
		}
		return result;
	}
	
	/**
	 * 
	 */
	public int[] voteLabel(int[][] y) {
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
		
		List<Integer> list = new ArrayList<Integer>();
		double n = (double)y.length / 2;
		
		Set<Integer> set = map.keySet();
		Iterator<Integer> it = set.iterator();
		while(it.hasNext()) {
			key = it.next();
			value = map.get(key);
			if(value > n) {
				list.add(key);
			}
		}
		
		int[] result = new int[list.size()];
		for(int i = 0; i < result.length; i++) {
			result[i] = list.get(i);
		}
		return result;
	}

	/**
	 * 
	 */
	public int[] doubleArrayToInt(double[] a) {
		if(a == null) {
			return null;
		}
		
		int[] result = new int[a.length];
		for(int i = 0; i < a.length; i++) {
			result[i] = (int)a[i];
		}
		return result;
	}
	
	/**
	 * 
	 */
	public int[][] doubleArrayToIntMat(double[] y) {
		if(y == null) {
			return null;
		}
		
		int[][] result = new int[y.length][1];
		for(int i = 0; i < y.length; i++) {
			result[i][0] = (int)y[i];
		}
		return result;
	}
	
	/**
	 * 
	 */
	public int[][] predictNear(double[][] pv, double[][] testpv, int[][] y, int k) {
		int[][] result = new int[testpv.length][];
		
		for(int i = 0; i < result.length; i++) {
			result[i] = getNearestLabel(pv, testpv[i], y, k);
		}
		
		return result;
	}
	
	/**
	 *	 
	 */
	public int[] getNearestLabel(double[][] pv, double[] testpv, int[][] y, int n) {
		double inner = 0;
		
		double[] sub = null;
		double[] distance = new double[pv.length];
		for(int i = 0; i < pv.length; i++) {
			
			sub = SparseVector.subVector(pv[i], testpv);
			
			inner = SparseVector.innerProduct(sub, sub);
			
			distance[i] = inner;
		}
		
		int[] index = Sort.getIndexBeforeSort(distance);
		
		int[][] pre = new int[n][];
		for(int i = 0; i < n; i++) {
			pre[i] = y[index[i]];
		}
		
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		for(int i = 0; i < pre.length; i++) {
			for(int j = 0; j < pre[i].length; j++) {
				if(map.containsKey(pre[i][j])) {
					int value = map.get(pre[i][j]);
					value = value + 1;
					map.put(pre[i][j], value);
				} else {
					map.put(pre[i][j], 1);
				}
			}
		}
		
		Set<Integer> set = map.keySet();
		Iterator<Integer> it = set.iterator();
		int key;
		int value;
		
		List<Integer> list = new ArrayList<Integer>();
		while(it.hasNext()) {
			key = it.next();
			value = map.get(key);
			if(value > ((double)n / 2)) {
				list.add(key);
			}
		}
		
		int[] result = new int[list.size()];
		for(int i = 0; i < result.length; i++) {
			result[i] = list.get(i);
		}
		return result;
	}

	/**
	 * 
	 */
	public void crossValidation(svm_problem prob, svm_parameter param, int n_fold, double c, double gamma) {
		int[] labels = getUniqueLabels(prob.y);
		
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
			
			svm_problem train = new svm_problem();
			train.l = trainIndex.length;
			train.x = new svm_node[train.l][];
			train.y = new double[train.l];
			
			counter = 0;
			for(int j = 0; j < trainIndex.length; j++) {
				train.x[counter] = prob.x[trainIndex[j]];
				train.y[counter] = prob.y[trainIndex[j]];
				counter++;
			}
			
			svm_problem valid = new svm_problem(); 
			valid.l = validIndex.length;
			valid.x = new svm_node[valid.l][];
			valid.y = new double[valid.l];
			
			counter = 0;
			for(int j = 0; j < validIndex.length; j++) {
				valid.x[counter] = prob.x[validIndex[j]];
				valid.y[counter] = prob.y[validIndex[j]];
				counter++;
			}
			
			param.C = c;
			param.gamma = gamma;
			
			
			svm_model[] models = train(train, param, labels);
			
			double[][] validpv = predict_values(valid, models);
			int[][] validPre = predict(validpv, labels);
			
			for(int k = 0; k < validPre.length; k++) {
				pre[validIndex[k]] = validPre[k];
			}
		}
		
		int[][] y = doubleArrayToIntMat(prob.y);
		double microf1 = Measures.microf1(labels, y, pre);
		double macrof1 = Measures.macrof1(labels, y, pre);
		double zeroneloss = Measures.zeroOneLoss(y, pre);
		System.out.println("gamma = " + gamma + ", C = " + c + 
				", Micro-F1 = " + microf1 + ", Macro-F1 = " + macrof1
				+", 0/1 loss = " + zeroneloss);
		
	}
}
