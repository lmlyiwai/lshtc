package com.threshold;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Linear;
import com.rssvm.Measures;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;
import com.tools.Contain;
import com.tools.Sort;

public class Scutfbr {
	
	/**
	 * 
	 * */
	public static Map<String, Object> scutfbr(Problem prob, int[] y, double[] fbr_list, Parameter param) {
		double[] b_list = new double[fbr_list.length];
		int nr_fold = 3;
		int l = y.length;
		int[] perm = Sort.readperm(l);
		double scut_b = 0;
		
		int seglength = l / nr_fold;
		for(int i = 0; i < nr_fold; i++) {
			int[] trainId = getTrainId(l, seglength, i);
			int[] validId = getValidId(l, seglength, i);
			
			trainId = transIndex(perm, trainId);
			validId = transIndex(perm, validId);
			
			int[] trainLabel = getSomeLabels(y, trainId);
			int[] validLabel = getSomeLabels(y, validId);
			
			Problem train = getSubProblem(prob, trainId);
			Problem valid = getSubProblem(prob, validId);
			
			DataPoint[] w = Linear.revisedTrain(train, trainLabel, param);
			
			double[] validPrdcit = predict(valid, w);
			
			int[] vy = getBinaryPredict(validPrdcit, 0);
			
			double startf = fmeasure(validLabel, vy);
			
			int[] vindex = Sort.getIndexBeforeSort(validPrdcit);
			
			double tp = getNumOfPositive(validLabel);
			double fp = validLabel.length - tp;
			double fn = 0;
			
			int cut = -1;
			double bestf = (2 * tp) / (2 * tp + fp + fn);
			double f;
			for(int j = 0; j < validLabel.length; j++) {
				if(validLabel[vindex[j]] == -1) {
					fp = fp - 1;
				} else {
					tp = tp - 1;
					fn = fn + 1;
				}
				
				f = (2 * tp) / (2 * tp + fp + fn);
				
				if(f > bestf) {
					bestf = f;
					cut = j;
				}
			}
			
			if(bestf > startf) {
				if(cut == -1) {
					scut_b = - (validPrdcit[vindex[0]] - 0.001);
				} else if (cut == (validLabel.length - 1)) {
					scut_b = - (validPrdcit[vindex[validPrdcit.length - 1]] + 0.001);
				} else {
					scut_b = - (validPrdcit[vindex[cut]] + validPrdcit[vindex[cut+1]]) / 2;
				}
			}
			
			vy = getBinaryPredict(validPrdcit, scut_b);
			f = fmeasure(validLabel, vy);
			
			for(int k = 0; k < fbr_list.length; k++) {
				if( f > fbr_list[k]) {
					b_list[k] += scut_b;
				} else {
					b_list[k] = b_list[k] - getMax(validPrdcit);
				}
			}
		}
		
		for(int i = 0; i < b_list.length; i++) {
			b_list[i] /= nr_fold;
		}
		
		DataPoint[] w = Linear.revisedTrain(prob, y, param);
		
		Map<String, Object> map = new HashMap<String, Object>();
		map.put("weight", w);
		map.put("b", b_list);
		
		return map;
	}
	
	/**
	 * 总长为totleLength, 每段长度seglength, 第k个确认集, k从0开始
	 * */
	public static int[] getValidId(int totleLength, int seglength, int k) {
		int start = k * seglength;
		int end = (k + 1) * seglength - 1;
		
		int[] result = new int[end - start + 1];
		for(int i = 0; i < result.length; i++) {
			result[i] = start + i;
		}
		return result;
	}
	
	public static int[] getTrainId(int totleLength, int seglength, int k) {
		int firststart = 0;
		int firstend = k * seglength - 1;
		int secondstart = (k + 1) * seglength;
		int secondend = totleLength - 1;
				
		List<Integer> list = new ArrayList<Integer>();
		for(int i = firststart; i <= firstend; i++) {
			list.add(i);
		}
		
		for(int i = secondstart; i <= secondend; i++) {
			list.add(i);
		}
		
		int[] result = new int[list.size()];
		for(int i = 0; i < result.length; i++) {
			result[i] = list.get(i);
		}
		return result;
	}
	
	/**
	 * 获得指定的标签
	 * */
	public static int[] getSomeLabels(int[] y, int[] index) {
		int[] result = new int[index.length];
		for(int i = 0; i < index.length; i++) {
			result[i] = y[index[i]];
		}
		return result;
	}
	
	/**
	 * 获得部分训练样本
	 * */
	public static Problem getSubProblem(Problem prob, int[] index) {
		Problem result = new Problem();
		result.bias = prob.bias;
		result.l = index.length;
		result.n = prob.n;
		result.x = new DataPoint[index.length][];
		
		for(int i = 0; i < index.length; i++) {
			result.x[i] = prob.x[index[i]];
		}
		return result;
	}
	
	/**
	 * 
	 * */
	public static double[] predict(Problem prob, DataPoint[] w) {
		double[] weight = SparseVector.sparseVectorToArray(w, prob.n);
		double[] result = new double[prob.l];
		for(int i = 0; i < result.length; i++) {
			result[i] = SparseVector.innerProduct(weight, prob.x[i]);
		}
		return result;
	}
	
	/**
	 * 
	 */
	public static int[] getBinaryPredict(double[] pv, double b) {
		int[] result = new int[pv.length];
		for(int i = 0; i < result.length; i++) {
			if(pv[i] + b > 0) {
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
	public static double fmeasure(int[] y, int[] predict) {
		double tp = Measures.truePositive(y, predict);
		double fp = Measures.falsePositive(y, predict);
		double fn = Measures.falseNegative(y, predict);
		
		
		double result = 0;
		if(tp != 0 || fp != 0 || fn != 0) {
			result = (2 * tp) / (2 * tp + fp + fn);
		}
		return result;
	}
	
	/**
	 * 
	 */
	public static double getNumOfPositive(int[] y) {
		double counter = 0;
		for(int i = 0; i < y.length; i++) {
			if(y[i] == 1) {
				counter++;
			}
		}
		return counter;
	}
	
	/**
	 * 
	 */
	public static double getMax(double[] a) {
		double max = Double.NEGATIVE_INFINITY;
		for(int i = 0; i < a.length; i++) {
			if(a[i] > max) {
				max = a[i];
			}
		}
		return max;
	}
	
	public static double getMin(double[] a) {
		double min = Double.POSITIVE_INFINITY;
		for(int i = 0; i < a.length; i++) {
			if(a[i] < min) {
				min = a[i];
			}
		}
		return min;
	}
	
	/**
	 * 
	 */
	public static Map<String, Object> trainOneLabel(Problem prob, int[] y, Parameter param) {
		double[] fbr_list = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
		int nr_fold = 3;
		int l = y.length;
		int seglength = l / nr_fold; 
		
		int[] perm = Sort.readperm(l);
		double[] f_list = new double[fbr_list.length];
		
		for(int i = 0; i < nr_fold; i++) {
			int[] trainId = getTrainId(l, seglength, i);
			int[] validId = getValidId(l, seglength, i);
			
			trainId = transIndex(perm, trainId);
			validId = transIndex(perm, validId);
			
			int[] trainLabel = getSomeLabels(y, trainId);
			int[] validLabel = getSomeLabels(y, validId);
			
			Problem trainprob = getSubProblem(prob, trainId);
			Problem validprob = getSubProblem(prob, validId);
			
			Map<String, Object> map = scutfbr(trainprob, trainLabel, fbr_list, param);
			DataPoint[] w = (DataPoint[])map.get("weight");
			double[] scutfbr_b_list = (double[])map.get("b");
			
			double[] validpredict = predict(validprob, w);
			
			for(int j = 0; j < fbr_list.length; j++) {
				int[] v = getBinaryPredict(validpredict, scutfbr_b_list[j]);
				double f = fmeasure(validLabel, v);
				f_list[j] = f_list[j] + f;
			}
		}
		
		int ind = getMaxIndex(f_list);
		double best_fbr = fbr_list[ind];
		
		if(getMax(f_list) == 0) {
			best_fbr = getMin(fbr_list);
		}
		
		double[] fbr = new double[1];
		fbr[0] = best_fbr;
		Map<String, Object> finalMap = scutfbr(prob, y, fbr, param);
		return finalMap;
	}
	
	/**
	 *	返回数组中最大值所在位置 
	 */
	public static int getMaxIndex(double[] array) {
		if(array == null) {
			return -1;
		}
		
		double max = Double.NEGATIVE_INFINITY;
		int index = -1;
		for(int i = 0; i < array.length; i++) {
			if(array[i] >= max) {
				max = array[i];
				index = i;
			}
		}
		
		return index;
	}
	
	/**
	 * 把顺序下标变为随机下标
	 */
	public static int[] transIndex(int[] perm, int[] index) {
		int[] result = new int[index.length];
		for(int i = 0; i < index.length; i++) {
			result[i] = perm[index[i]];
		}
		return result;
	}
	
	/**
	 * labels所有标签，pv预测输出值，y实际标签
	 */
	public static double[] getThreshold(int[] labels, double[][] pv, int[][] y) {
		int i, j;
		int label;
		int[] yi;
		double[] pyi;
		
		double[] t = new double[labels.length];
		
		for(i = 0; i < labels.length; i++) {
			label = labels[i];
			yi = new int[y.length];
			for(j = 0; j < yi.length; j++) {
				if(Contain.contain(y[j], label)) {
					yi[j] = 1;
				} else {
					yi[j] = -1;
				}
			}
			
			pyi = getKColumn(pv, i);
			
			t[i] = getT(pyi, yi);
		}
		return t;
	}
	
	/**
	 * 获得二维数组第k列
	 */
	public static double[] getKColumn(double[][] pv, int k) {
		if(pv == null) {
			return null;
		}
		
		double[] result = new double[pv.length];
		for(int i = 0; i < result.length; i++) {
			result[i] = pv[i][k];
		}
		return result;
	}
	
	/**
	 *	获得最佳threshold 
	 */
	public static double getT(double[] pv, int[] y) {
		double tp = getNumOfPositive(y);
		double fp = y.length - tp;
		double fn = 0;
		double startf = (2 * tp) / (2 * tp + fp + fn);
		double bestf = (2 * tp) / (2 * tp + fp + fn);
		
		int cut = -1;
		double f;
		int[] sortIndex = Sort.getIndexBeforeSort(pv);
		for(int i = 0; i < pv.length; i++) {
			if(y[sortIndex[i]] == -1) {
				fp = fp - 1;
			}  else {
				tp = tp - 1;
				fn = fn + 1;
			}
			
			f = (2 * tp) / (2 * tp + fp + fn);
			
			if(f > bestf) {
				bestf = f;
				cut = i;
			}
		}
		
		double scut = 0;
		if(bestf > startf) {
			if(cut == -1) {
				scut = pv[sortIndex[0]] - 0.5;
			} else if (cut == y.length - 1) {
				scut = pv[sortIndex[y.length - 1]] + 0.5;
			} else {
				scut = (pv[sortIndex[cut]] + pv[sortIndex[cut+1]]) / 2;
			}
		}
		
		return scut;
	}
	
}
