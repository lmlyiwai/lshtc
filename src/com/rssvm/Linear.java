package com.rssvm;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import javax.xml.crypto.Data;

import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;
import com.tools.RandomSequence;
import com.tools.Sort;



public class Linear {
	private static Random random = new Random();
	
	public static DataPoint[] train(Problem prob, int[] y, Parameter param, DataPoint[] parent, double[] tloss, DataPoint[][] alpha_id, int id) {
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
		
		double[] parentW = dpToArray(parent, prob.n);
		double[] w = new double[prob.n];
		
		for(i = 0; i < l; i++) {
			alpha[i] = 0;
		}
		
		if(alpha_id != null && alpha_id[id] != null) {                                  //上次训练结束后alpha值
			for(DataPoint dp : alpha_id[id]) {
				alpha[dp.index - 1] = dp.value;
			}
		}
		
		for(i = 0; i < w_size; i++) {
			w[i] = parentW[i];
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
		
		double loss = 0;
		double output;
		DataPoint[] ti;
		int yi;
		for(i = 0; i < l; i++) {
			output = 0;
			ti = prob.x[i];
			yi = y[i];
			for(DataPoint dp : ti) {
				output += w[dp.index - 1] * dp.value;
			}
			
			loss += Math.max(0, 1 - yi * output);
		}
		tloss[0] = loss;
		
		
		List<DataPoint> list = new ArrayList<DataPoint>();			//保存每个样本对应的alpha
		DataPoint temp;
		for(i = 0; i < l; i++) {
			if(alpha[i] != 0) {
				temp = new DataPoint(i+1, alpha[i]);
				list.add(temp);
			}
		}
		
		if(alpha_id != null) {
			alpha_id[id] = new DataPoint[list.size()];
			for(i = 0; i < alpha_id[id].length; i++) {
				alpha_id[id][i] = list.get(i);
			}
		}
		
		return arrToVec(w);
	}
	
	/**
	 * 将稀疏向量变为全向量
	 * */
	public static double[] dpToArray(DataPoint[] v, int n) {
		double[] result = new double[n];
		for(int i = 0; i < result.length; i++) {
			result[i] = 0;
		}
		
		if(v == null) {
			return result;
		}
		
		for(DataPoint p : v) {
			result[p.index - 1] = p.value;
		}
		return result;
	}
	
	public static void swap(int[] index, int i, int j) {
		int temp = index[i];
		index[i] = index[j];
		index[j] = temp;
	}
	
	public static DataPoint[] arrToVec(double[] w) {
		List<DataPoint> list = new ArrayList<DataPoint>();
		for(int i = 0; i < w.length; i++) {
			if(w[i] != 0.0) {
				list.add(new DataPoint(i+1, w[i]));
			}
		}
		
		DataPoint[] result = new DataPoint[list.size()];
		for(int i = 0; i < result.length; i++) {
			result[i] = list.get(i);
		}
		return result;
	}
	
	/**
	 * 增加权值之后，叶节点训练。
	 * */
	public static DataPoint[] revisedTrain(Problem prob, int[] y, Parameter param, DataPoint[] parent, double[] tloss, DataPoint[][] alpha_id, int id, double B) {
		int l 			= prob.l;
		int w_size 		= prob.n;
		int[] index 	= new int[l];
		double[] alpha 	= new double[l];
		int active_size = l;
		int i, s, iter 	= 0;
		double C, d, G, it1, it2;
		double[] QD 	= new double[l];
		
		double PG;
		double PGmax_old = Double.POSITIVE_INFINITY;
		double PGmin_old = Double.NEGATIVE_INFINITY;
		double PGmax_new, PGmin_new;
		
		double[] parentW = dpToArray(parent, prob.n);
		double[] w = new double[prob.n];
		double[] a = new double[prob.n];
		
		for(i = 0; i < l; i++) {
			alpha[i] = 0;
		}
		
		if(alpha_id != null && alpha_id[id] != null) {                                  //上次训练结束后alpha值
			for(DataPoint dp : alpha_id[id]) {
				alpha[dp.index - 1] = dp.value;
			}
		}
		
		for(i = 0; i < w_size; i++) {
			w[i] = (1 + 2 * B) * parentW[i];
		}
		
		for(i = 0; i < l; i++) {
			QD[i] = 0;
			for(DataPoint dp : prob.x[i]) {
				double val = dp.value;
				QD[i] += (val * val * (1 / (1 + 2 * B)));
				w[dp.index - 1] += (1 + 2 * B) * y[i] * alpha[i] * val;
				a[dp.index - 1] += y[i] * alpha[i] * val;
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
				
				it1 = 0; it2 = 0;
				for(DataPoint xi : prob.x[i]) {
					it1 += a[xi.index - 1] * xi.value;
					it2 += parentW[xi.index - 1] * xi.value;
				}
				
				
				G = (1 / (1 + 2 * B)) * yi * it1 + ((1 - 4 * B) / ((1 + 2 * B) * (1 + 2 * B))) * yi * it2 - 1;
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
						a[xi.index - 1] += d * xi.value;
						w[xi.index - 1] += d * xi.value * (1 / (1 + 2 * B));
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
		
		double loss = 0;
		double output;
		DataPoint[] ti;
		int yi;
		for(i = 0; i < l; i++) {
			output = 0;
			ti = prob.x[i];
			yi = y[i];
			for(DataPoint dp : ti) {
				output += w[dp.index - 1] * dp.value;
			}
			
			loss += Math.max(0, 1 - yi * output);
		}
		tloss[0] = loss;
		
		
		List<DataPoint> list = new ArrayList<DataPoint>();			//保存每个样本对应的alpha
		DataPoint temp;
		for(i = 0; i < l; i++) {
			if(alpha[i] != 0) {
				temp = new DataPoint(i+1, alpha[i]);
				list.add(temp);
			}
		}
		
		if(alpha_id != null) {
			alpha_id[id] = new DataPoint[list.size()];
			for(i = 0; i < alpha_id[id].length; i++) {
				alpha_id[id][i] = list.get(i);
			}
		}
		
		return arrToVec(w);
	}
	
	public static int numOfPositive(int[] y) {
		int counter = 0;
		for(int i = 0; i < y.length; i++) {
			if(y[i] == 1) {
				counter++;
			}
		}
		
		return counter;
	}
	
	public static DataPoint[] revisedTrain(Problem prob, int[] y, Parameter param) {
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
		
		double[] w = new double[prob.n];
		
		for(i = 0; i < l; i++) {
			alpha[i] = 0;
		}
		
		
		for(i = 0; i < w_size; i++) {
			w[i] = 0;
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
		
		double loss = 0;
		double output;
		DataPoint[] ti;
		int yi;
		for(i = 0; i < l; i++) {
			output = 0;
			ti = prob.x[i];
			yi = y[i];
			for(DataPoint dp : ti) {
				output += w[dp.index - 1] * dp.value;
			}
			
			loss += Math.max(0, 1 - yi * output);
		}
		
		List<DataPoint> list = new ArrayList<DataPoint>();			//保存每个样本对应的alpha
		DataPoint temp;
		for(i = 0; i < l; i++) {
			if(alpha[i] != 0) {
				temp = new DataPoint(i+1, alpha[i]);
				list.add(temp);
			}
		}	
		return arrToVec(w);
	}

	/**
	 *  正则项系数nd
	 */
	public static DataPoint[] train(Problem prob, int[] y, Parameter param, double nd) {
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
		double[] w = new double[prob.n];
		
		for(i = 0; i < l; i++) {
			alpha[i] = 0;
		}
			
		for(i = 0; i < w_size; i++) {
			w[i] = 0;
		}
		
		for(i = 0; i < l; i++) {
			QD[i] = 0;
			for(DataPoint dp : prob.x[i]) {
				double val = dp.value;
				QD[i] += val * val / (2 * nd);
				w[dp.index - 1] += (1 / (2 * nd)) * y[i] * alpha[i] * val;
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
						w[xi.index - 1] += d * xi.value / (2 * nd);
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
		
		return arrToVec(w);
	}
	
	/**
	 * 增加cost 
	 */
	public static DataPoint[] train(Problem prob, int[] y, Parameter param, DataPoint[] parent, double[] cost, double[] tloss, DataPoint[][] alpha_id, int id) {
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
		
		double[] parentW = dpToArray(parent, prob.n);
		double[] w = new double[prob.n];
		
		for(i = 0; i < l; i++) {
			alpha[i] = 0;
		}
		
		if(alpha_id != null && alpha_id[id] != null) {                                  //上次训练结束后alpha值
			for(DataPoint dp : alpha_id[id]) {
				alpha[dp.index - 1] = dp.value;
			}
		}
		
		for(i = 0; i < w_size; i++) {
			w[i] = parentW[i];
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
				double cos = cost[i];		//样本i对应损失代价
				
				for(DataPoint xi : prob.x[i]) {
					G += w[xi.index - 1] * xi.value;
				}
				
				G = G * yi - 1;
				C = param.getC() * cos;         //alpha[i]变化范围
				
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
		
		double loss = 0;
		double output;
		DataPoint[] ti;
		int yi;
		for(i = 0; i < l; i++) {
			output = 0;
			ti = prob.x[i];
			yi = y[i];
			for(DataPoint dp : ti) {
				output += w[dp.index - 1] * dp.value;
			}
			
			loss += Math.max(0, 1 - yi * output);
		}
		tloss[0] = loss;
		
		
		List<DataPoint> list = new ArrayList<DataPoint>();			//保存每个样本对应的alpha
		DataPoint temp;
		for(i = 0; i < l; i++) {
			if(alpha[i] != 0) {
				temp = new DataPoint(i+1, alpha[i]);
				list.add(temp);
			}
		}
		
		if(alpha_id != null) {
			alpha_id[id] = new DataPoint[list.size()];
			for(i = 0; i < alpha_id[id].length; i++) {
				alpha_id[id][i] = list.get(i);
			}
		}
		
		return arrToVec(w);
	}
	
	public static DataPoint[] train(Problem prob, int[] y, Parameter param, double[] margin) {
		int l 			= prob.l;
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
		
		double[] w = new double[prob.n];
		
		for(i = 0; i < l; i++) {
			alpha[i] = 0;
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
				double mar = margin[i];
				
				for(DataPoint xi : prob.x[i]) {
					G += w[xi.index - 1] * xi.value;
				}
				
				G = G * yi - mar;
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
		return arrToVec(w);
	}
	
	public static DataPoint[] train(Problem prob, Parameter param, double[] margin) {
		int l 			= prob.l;
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
		
		double[] w = new double[prob.n];
		
		for(i = 0; i < l; i++) {
			alpha[i] = 0;
		}
		
		for(i = 0; i < l; i++) {
			QD[i] = 0;
			for(DataPoint dp : prob.x[i]) {
				double val = dp.value;
				QD[i] += val * val;
				w[dp.index - 1] += alpha[i] * val;
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
				double mar = margin[i];
				
				for(DataPoint xi : prob.x[i]) {
					G += w[xi.index - 1] * xi.value;
				}
				
				G = G - mar;
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
					d = alpha[i] - alpha_old;
					
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
		return arrToVec(w);
	}
	
	/**
	 *    min  0.5 * w ^ 2 + c * sum(e_i)
	 *    s.t. y_i * w * x_i >= 1 - e_i
	 *         y_i * w_p * x_i >= y_i * w * x_i 
	 */
	public static DataPoint[] train(Problem prob, int[] y, Parameter param, DataPoint[] parent) {
		int l = prob.l;
		int wsize = prob.n;
		int[] index = new int[l];
		double[] alpha = new double[l];
		double[] r = new double[l];
		double[] w = new double[wsize];
		double[] QD = new double[l];
		
		for(int i = 0; i < l; i++) {
			alpha[i] = 0;
			r[i] = 0;
		}
		
		for(int i =0; i < l; i++) {
			for(DataPoint dp : prob.x[i]) {
				w[dp.index - 1] += alpha[i] * y[i] * dp.value;
				w[dp.index - 1] -= r[i] * y[i] * dp.value;
			}
			
			QD[i] = 0;
			for(DataPoint dp : prob.x[i]) {
				QD[i] += dp.value * dp.value;
			}
		}
		
		int counter = 0;
		while(counter < param.getMaxIteration()) {
			index = RandomSequence.randomSequence(l);
			for(int i = 0; i < l; i++) {
				double alpha_old = alpha[index[i]];
				int yi = y[index[i]];
				DataPoint[] x = prob.x[index[i]];
				double fenmu = 1 - yi * SparseVector.innerProduct(w, x);
				double d = fenmu / QD[index[i]];
				double alpha_new = alpha_old + d;
				alpha_new = Math.max(Math.min(alpha_new, param.getC()), 0);
				alpha[index[i]] = alpha_new;
				d = alpha_new - alpha_old;
				for(DataPoint dp : x) {
					w[dp.index - 1] += d * yi * dp.value;
				}
			}
			
			index = RandomSequence.randomSequence(l);
			for(int i = 0; i < l; i++) {
				double r_old = r[index[i]];
				int yi = y[index[i]];
				DataPoint[] x = prob.x[index[i]];
				double fenmu = yi * SparseVector.innerProduct(w, x) - yi * SparseVector.innerProduct(parent, x);
				double d = fenmu / QD[index[i]];
				double r_new = r_old + d;
				r_new = Math.max(r_new, 0);
				r[index[i]] = r_new;
				d = r_old - r_new;
				for(DataPoint dp : x) {
					w[dp.index - 1] += d * yi * dp.value;
				}
			}
			
			if(counter != 0 && counter % 10 == 0) {
				double loss = 0;
				for(int i = 0; i < l; i++) {
					double tloss = 1 - y[i] * SparseVector.innerProduct(w, prob.x[i]);
					loss += Math.max(0, tloss);
				}
				double primalObj = 0.5 * SparseVector.innerProduct(w, w) + loss;
				
				double alphaSum = 0;
				double t = 0;
				for(int i = 0; i < l; i++) {
					alphaSum += alpha[i];
					t += r[i] * y[i] * SparseVector.innerProduct(parent, prob.x[i]);
				}
				
				double dualObj = alphaSum - 0.5 * SparseVector.innerProduct(w, w) - t;
				
				System.out.println("Primal Obj = " + primalObj + ", Dual Obj = " + dualObj);
				if(primalObj - dualObj <= param.getEps()) {
					break;
				}
			}
			
			counter = counter + 1;
		}
		
		DataPoint[] result = arrToVec(w);
		return result;
	}
	
	/**
	 *    min  0.5 * w ^ 2 + c * sum(e_i)
	 *    s.t. y_i * w * x_i >= 1 - e_i
	 *         y_i * w_p * x_i >= y_i * w * x_i 
	 */
	public static DataPoint[] newtrain(Problem prob, int[] y, Parameter param, DataPoint[] parent) {
		int l = prob.l;
		int wsize = prob.n;
		int[] index = new int[l];
		double[] alpha = new double[l];
		double[] r = new double[l];
		double[] w = new double[wsize];
		double[] QD = new double[l];
		
		for(int i = 0; i < l; i++) {
			alpha[i] = 0;
			r[i] = 0;
		}
		
		for(int i =0; i < l; i++) {
			for(DataPoint dp : prob.x[i]) {
				w[dp.index - 1] += alpha[i] * y[i] * dp.value;
				w[dp.index - 1] -= r[i] * y[i] * dp.value;
			}
			
			QD[i] = 0;
			for(DataPoint dp : prob.x[i]) {
				QD[i] += dp.value * dp.value;
			}
		}
		
		int counter = 0;
		while(counter < param.getMaxIteration()) {
			index = RandomSequence.randomSequence(l);
			for(int i = 0; i < l; i++) {
				double alpha_old = alpha[index[i]];
				int yi = y[index[i]];
				DataPoint[] x = prob.x[index[i]];
				double fenmu = 1 - yi * SparseVector.innerProduct(w, x);
				double d = fenmu / QD[index[i]];
				double alpha_new = alpha_old + d;
				alpha_new = Math.max(Math.min(alpha_new, param.getC()), 0);
				alpha[index[i]] = alpha_new;
				d = alpha_new - alpha_old;
				for(DataPoint dp : x) {
					w[dp.index - 1] += d * yi * dp.value;
				}
				
				double r_old = r[index[i]];
				fenmu = yi * SparseVector.innerProduct(w, x) - yi * SparseVector.innerProduct(parent, x);
				d = fenmu / QD[index[i]];
				double r_new = r_old + d;
				r_new = Math.max(Math.min(r_new, param.getC1()), 0);
				r[index[i]] = r_new;
				d = r_old - r_new;
				
				for(DataPoint dp : x) {
					w[dp.index - 1] += d * yi * dp.value;
				}
			}
			
			if(counter != 0 && counter % 10 == 0) {
				double loss = 0;
				double loss1 = 0;
				for(int i = 0; i < l; i++) {
					double tloss = 1 - y[i] * SparseVector.innerProduct(w, prob.x[i]);
					loss += Math.max(0, tloss);
					
					double tloss1 = y[i] * SparseVector.innerProduct(w, prob.x[i]) -
									y[i] * SparseVector.innerProduct(parent, prob.x[i]);
					loss1 += Math.max(0, tloss1);
				}
				double primalObj = 0.5 * SparseVector.innerProduct(w, w) + param.getC() * loss
						+ param.getC1() * loss1;
				
				double alphaSum = 0;
				double t = 0;
				for(int i = 0; i < l; i++) {
					alphaSum += alpha[i];
					t += r[i] * y[i] * SparseVector.innerProduct(parent, prob.x[i]);
				}
				
				double dualObj = alphaSum - 0.5 * SparseVector.innerProduct(w, w) - t;
				
//				System.out.println("Primal Obj = " + primalObj + ", Dual Obj = " + dualObj);
				if((primalObj - dualObj) / ((primalObj + dualObj) / 2) <= param.getEps()) {
					break;
				}
			}
			
			counter = counter + 1;
		}
		
		DataPoint[] result = arrToVec(w);
		return result;
	}
	
	public static DataPoint[] train(double[][] pv, int[] y, Parameter param, DataPoint[] parent, double[] tloss, DataPoint[][] alpha_id, int id) {
		int l 			= pv.length;
		int w_size 		= pv[0].length;
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
		
		double[] parentW = dpToArray(parent, pv[0].length);
		double[] w = new double[pv[0].length];
		
		for(i = 0; i < l; i++) {
			alpha[i] = 0;
		}
		
		if(alpha_id != null && alpha_id[id] != null) {                                  //上次训练结束后alpha值
			for(DataPoint dp : alpha_id[id]) {
				alpha[dp.index - 1] = dp.value;
			}
		}
		
		for(i = 0; i < w_size; i++) {
			w[i] = parentW[i];
		}
		
		for(i = 0; i < l; i++) {
			QD[i] = SparseVector.innerProduct(pv[i], pv[i]);
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
				
				G = SparseVector.innerProduct(w, pv[i]);
				
				G = G * yi - 1;
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
					
//					for(DataPoint xi : prob.x[i]) {
//						w[xi.index - 1] += d * xi.value;
//					}
					
					double[] dv = SparseVector.scaleVector(pv[i], d);
					SparseVector.localVecAdd(w, dv);
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
		
		double loss = 0;
		double output;
		int yi;
		for(i = 0; i < l; i++) {
			output = 0;
			yi = y[i];
			output = SparseVector.innerProduct(w, pv[i]);
			loss += Math.max(0, 1 - yi * output);
		}
		tloss[0] = loss;
		
		
		List<DataPoint> list = new ArrayList<DataPoint>();			//保存每个样本对应的alpha
		DataPoint temp;
		for(i = 0; i < l; i++) {
			if(alpha[i] != 0) {
				temp = new DataPoint(i+1, alpha[i]);
				list.add(temp);
			}
		}
		
		if(alpha_id != null) {
			alpha_id[id] = new DataPoint[list.size()];
			for(i = 0; i < alpha_id[id].length; i++) {
				alpha_id[id][i] = list.get(i);
			}
		}
		
		return arrToVec(w);
	}
}
