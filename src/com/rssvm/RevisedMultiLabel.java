package com.rssvm;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;
import com.tools.Contain;
import com.tools.Statistics;

public class RevisedMultiLabel {
	private Problem 	prob;
	private int[] 		labels;
	private double[][] 	w;
	private double[][] 	alphas;
	private Parameter 	param;
	private static Random random = new Random();
	
	public RevisedMultiLabel(Problem prob, Parameter param) {
		this.prob = prob;
		this.param = param;
		this.labels = Statistics.getUniqueLabels(prob.y);
	}
	
	public void train() {
		this.w = new double[this.labels.length][];
		this.alphas = new double[this.labels.length][this.prob.l];
		
		int label;
		int j;
		int[] y;
		double[] bound = null;
		
		int tc = 0;
		double norm = 0;
		double lastNorm = 0;
		while(tc < 1000) {
			System.out.print(tc + ", ");
			norm = 0;
			for(int i = 0; i < this.labels.length; i++) {
				label = this.labels[i];
				y = new int[this.prob.l];
				for(j = 0; j < y.length; j++) {
					if(Contain.contain(this.prob.y[j], label)) {
						y[j] = 1;
					} else {
						y[j] = -1;
					}
				}
				
				bound = getbound(this.alphas, i, this.param.getC());
				this.w[i] = train(prob, y, param, this.alphas[i], bound);
				norm += SparseVector.innerProduct(this.w[i], this.w[i]);
			}
			
			if(Math.abs(norm - lastNorm) < 0.01) {
				break;
			}
			lastNorm = norm;
			tc++;
		}
	}
	
	/**
	 * 
	 */
	public static double[] getbound(double[][] m, int r, double c) {
		if(m == null) {
			return null;
		}
		
		int length = m[0].length;
		double[] result = new double[length];
		int i, j;
		for(j = 0; j < length; j++) {
			for(i = 0; i < m.length; i++) {
				if(i != r) {
					result[j] += m[i][j];
				}
			}
		}
		
		for(i = 0; i < result.length; i++) {
			result[i] = c - result[i];
		}
		return result;
	}
	
	/**
	 * 
	 */
	public static double[] train(Problem prob, int[] y, Parameter param, double[] alphaOld, double[] bound) {
		int l 			= prob.l;
		int w_size 		= prob.n;
		int[] index 	= new int[l];
		double[] alpha 	= new double[l];
		int active_size = l;
		int i, s, iter 	= 0;
		double C, d, G;
		double[] QD 	= new double[l];
		
		double PG;
		double PGmax_old = Double.POSITIVE_INFINITY;
		double PGmin_old = Double.NEGATIVE_INFINITY;
		double PGmax_new, PGmin_new;
		

		double[] w = new double[w_size];
		
		for(i = 0; i < l; i++) {
			alpha[i] = alphaOld[i];
		}
		
	
		for(i = 0; i < l; i++) {
			QD[i] = 0;
			for(DataPoint dp : prob.x[i]) {
				double val = dp.value;
				QD[i] += val * val;
				w[dp.index - 1] += y[i] * alpha[i] * val;
			}
			index[i] = i;
		}
		
		while(iter < param.getMaxIteration()) {
			PGmax_new = Double.NEGATIVE_INFINITY;
			PGmin_new = Double.POSITIVE_INFINITY;
			
			for(i = 0; i < active_size; i++) {
				int j = i + random.nextInt(active_size - i);
				swap(index, i, j);
			}
			
			for(s = 0; s < active_size; s++) {
				i = index[s];
				G = 0;
				int yi = y[i];
				
				for(DataPoint xi : prob.x[i]) {
					G += w[xi.index - 1] * xi.value;
				}
				
				G = G * yi - 1;
				C = bound[i];
				
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
				} else if (alpha[i] == C) {
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
					alpha[i] = Math.min(Math.max((alpha[i] - (G / QD[i])), 0.0), C);
					d = (alpha[i] - alpha_old) * yi;
					
					for(DataPoint xi : prob.x[i]) {
						w[xi.index - 1] += d * xi.value;
					}
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
		
		
		for(i = 0; i < alpha.length; i++) {
			alphaOld[i] = alpha[i];
		}
		return w;
	}
	
	public static void swap(int[] index, int i, int j) {
		int temp = index[i];
		index[i] = index[j];
		index[j] = temp;
	}
	
	public int[][] predict(DataPoint[][] samples) {
		int[][] pre = new int[samples.length][];
		
		int i, j, counter = 0;
		DataPoint[] sample = null;
		double[] pv = new double[this.labels.length];
		for(i = 0; i < samples.length; i++) {
			sample = samples[i];
			counter = 0;
			for(j = 0; j < this.w.length; j++) {
				pv[j] = SparseVector.innerProduct(w[j], sample);
				if(pv[j] > 0) {
					counter++;
				}
			}
			
			pre[i] = new int[counter];
			counter = 0;
			for(j = 0; j < pv.length; j++) {
				if(pv[j] > 0) {
					pre[i][counter++] = this.labels[j];
				}
			}
		}
		
		return pre;
	}

	public int[] getLabels() {
		return labels;
	}

	public void setLabels(int[] labels) {
		this.labels = labels;
	}
	
}
