package com.nonlinear.RCV1;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import com.examples.svm;
import com.examples.svm_model;
import com.examples.svm_node;
import com.examples.svm_parameter;
import com.examples.svm_problem;
import com.fileInputOutput.Problem;
import com.rssvm.Measures;
import com.sparseVector.DataPoint;
import com.tools.Contain;
import com.tools.RandomSequence;

public class NonlinearRCV1 {
	/**
	 *	 
	 */
	public int[] getLabels(int[][] y) {
		if(y == null) {
			return null;
		}
		
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
	 * 
	 */
	public svm_problem transformProblem(Problem prob, int labels) {
		svm_problem sp = new svm_problem();
		sp.l = prob.l;
		sp.x = new svm_node[sp.l][];
		sp.y = new double[sp.l];
		
		DataPoint[] x = null;
		int[] y = null;
		for(int i = 0; i < prob.l; i++) {
			x = prob.x[i];
			y = prob.y[i];
			
			sp.x[i] = new svm_node[x.length];
			for(int j = 0; j < x.length; j++) {
				sp.x[i][j] = new svm_node();
				
				sp.x[i][j].index = x[j].index;
				sp.x[i][j].value = x[j].value;
			}
			if(Contain.contain(y, labels)) {
				sp.y[i] = 1;
			} else {
				sp.y[i] = -1;
			}
		}
		return sp;
	}
	
	/**
	 * 
	 */
	public svm_model[] train(Problem prob, double gamma, double c, int[] labels) {
		svm_model[] models = new svm_model[labels.length];
		svm_parameter param = setParameter(2, 3, gamma, c);
		
		for(int i = 0; i < labels.length; i++) {
			svm_problem sp = transformProblem(prob, labels[i]);
			models[i] = svm.svm_train(sp, param);
		}
		return models;
	}
	
	/**
	 * 
	 */
	public double[][] predictValues(svm_model[] models, svm_problem prob) {
		double[] pv = new double[1];
		double[][] predictv = new double[prob.l][models.length];
		for(int i = 0; i < models.length; i++) {
			for(int j = 0; j < prob.l; j++) {
				svm.svm_predict_values(models[i], prob.x[j], pv);
				predictv[j][i] = pv[0];
			}
		}
		return predictv;
	}
	
	/**
	 * 
	 */
	public int[][] predict(double[][] pv, int[] labels) {
		int[][] pl = new int[pv.length][];
		for(int i = 0; i < pl.length; i++) {
			pl[i] = predictSingle(pv[i], labels);
		}
		return pl;
	}
	
	/**
	 * 
	 */
	public int[] predictSingle(double[] pv, int[] labels) {
		List<Integer> list = new ArrayList<Integer>();
		for(int i = 0; i < pv.length; i++) {
			if(pv[i] > 0) {
				list.add(labels[i]);
			}
		}
		int[] result = new int[list.size()];
		for(int i = 0; i < result.length; i++) {
			result[i] = list.get(i);
		}
		return result;
	}
	
	public svm_parameter setParameter(int kernel_type, int degree, double gamma, double C) {
		svm_parameter param = new svm_parameter();
		// default values
		param.svm_type = svm_parameter.C_SVC;
		param.kernel_type = kernel_type;
		param.degree = degree;
		param.gamma = gamma;	// 1/num_features
		param.coef0 = 0;
		param.nu = 0.5;
		param.cache_size = 1000;
		param.C = C;
		param.eps = 1e-3;
		param.p = 0.1;
		param.shrinking = 1;
		param.probability = 0;
		param.nr_weight = 0;
		param.weight_label = new int[0];
		param.weight = new double[0];
		
		return param;
	}
	
	/**
	 *	Ò»¶Ô¶à 
	 */
	public void crossValidation(Problem prob, int n_fold, double gamma, double c) {
		int[] labels = getLabels(prob.y);
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
			
			svm_model[] models = train(train, gamma, c, labels);
			
			svm_problem vpro = transformProblem(valid, labels[0]);
			
			double[][] pv = predictValues(models, vpro);
			int[][] predictLabel = predict(pv, labels);
			for(int j = 0; j < validIndex.length; j++) {
				pre[validIndex[j]] = predictLabel[j];
			}
		}
		
		double microf1 = Measures.microf1(labels, prob.y, pre);
		double macrof1 = Measures.macrof1(labels, prob.y, pre);
		System.out.println("gamma = " + gamma + ", c = " + c 
				+ ", Micro-F1 = " + microf1 + ", Macro-F1 = " + macrof1);
	}
}
