package com.meta;

import java.util.ArrayList;
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
import com.tools.RandomSequence;
import com.tools.Sort;

public class MetaSvmKnn {
	private Problem 		prob;
	private Parameter 		param;
	private int[] 			uniqueLabels;
	private DataPoint[][] 	weight;
	private DataPoint[][] 	weight1;
	private Structure		structure;
	
	public MetaSvmKnn(Problem prob, Parameter param) {
		this.prob = prob;
		this.param = param;
		this.uniqueLabels = getAllLabels(prob.y);
	}
	
	
	public Structure getStructure() {
		return structure;
	}


	public void setStructure(Structure structure) {
		this.structure = structure;
	}


	public DataPoint[][] train(Problem prob, Parameter param) {
		DataPoint[][] w = new DataPoint[this.uniqueLabels.length][];
		int label;
		
		int[] y;
		double[] tloss = new double[1];
		for(int i = 0; i < this.uniqueLabels.length; i++) {
			label = this.uniqueLabels[i];
			y = getLabels(prob.y, label);
			w[i] = Linear.train(prob, y, param, null, tloss, null, 0);
		}
		return w;
	} 
	
	/**
	 *	获得训练集中出现的标签 
	 */
	public int[] getAllLabels(int[][] y) {
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
	 *	获得id指定类标签 
	 */
	public int[] getLabels(int[][] y, int label) {
		if(y == null) {
			return null;
		}
		
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
	 *	 预测输出值
	 */
	public double[][] predictValues(DataPoint[][] w, DataPoint[][] x) {
		int n = this.uniqueLabels.length;
		
		double[][] pv = new double[x.length][n];
		
		double[] weight;
		DataPoint[] tx;
		
		for(int i = 0; i < w.length; i++) {
			weight = SparseVector.sparseVectorToArray(w[i], this.prob.n);
			for(int j = 0; j < x.length; j++) {
				tx = x[j];
				pv[j][i] = SparseVector.innerProduct(weight, tx); 
			}
		}
		return pv;
	}

	/**
	 *	 预测输出值
	 */
	public double[][] predictValues(DataPoint[][] w, DataPoint[][] x, int length) {
		int n = this.uniqueLabels.length;
		
		double[][] pv = new double[x.length][n];
		
		double[] weight;
		DataPoint[] tx;
		
		for(int i = 0; i < w.length; i++) {
			weight = SparseVector.sparseVectorToArray(w[i], length);
			for(int j = 0; j < x.length; j++) {
				tx = x[j];
				pv[j][i] = SparseVector.innerProduct(weight, tx); 
			}
		}
		return pv;
	}
	
	/**
	 * 
	 */
	public double[][] predictValues(DataPoint[][] w, double[][] x) {
		int n = this.uniqueLabels.length;
		
		double[][] pv = new double[x.length][n];
		
		double[] weight;
		double[] tx;
		
		for(int i = 0; i < w.length; i++) {
			weight = SparseVector.sparseVectorToArray(w[i], this.prob.n);
			for(int j = 0; j < x.length; j++) {
				tx = x[j];
				pv[j][i] = SparseVector.innerProduct(weight, tx); 
			}
		}
		return pv;
	}
	
	
	/**
	 *	预测类别 
	 */
	public int[][] predict(double[][] pv) {
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
					result[i][counter++] = this.uniqueLabels[j];
				}
			}
		}
		return result;
	}

	/**
	 * 单类标预测 
	 */
	public int[][] predictMax(double[][] pv) {
		int index = -1;
		double max = Double.NEGATIVE_INFINITY;
		
		int[][] result = new int[pv.length][1];
		
		for(int i = 0; i < pv.length; i++) {
			index = -1;
			max = Double.NEGATIVE_INFINITY;
			for(int j = 0; j < pv[i].length; j++) {
				if(pv[i][j] > max) {
					max = pv[i][j];
					index = j;
				}
			}
			
			result[i] = new int[1];
			result[i][0] = this.uniqueLabels[index];
		}
		return result;
	}
	/**
	 *	全矩阵转换为DataPoint形式 
	 */
	public DataPoint[][] trans(double[][] pv, double bias) {
		DataPoint[][] result = new DataPoint[pv.length][];
		
		DataPoint temp;
		int index;
		double value;
		
		boolean flag = false;
		int length = pv[0].length;
		if(bias > 0) {
			length = length + 1;
			flag = true;
		}
		
		int counter = 0;
		for(int i = 0; i < pv.length; i++) {
			
			result[i] = new DataPoint[length];
			counter = 0;
			for(int j = 0; j < pv[i].length; j++) {
				index = j + 1;
				value = pv[i][j];
				temp = new DataPoint(index, value);
				result[i][counter++] = temp;
			}
			
			if(flag) {
				temp = new DataPoint(length, bias);
				result[i][counter] = temp;
			}
		}
		return result;
	}
	
	
	public int[] getUniqueLabels() {
		return uniqueLabels;
	}

	public void setUniqueLabels(int[] uniqueLabels) {
		this.uniqueLabels = uniqueLabels;
	}
	
	/**
	 *	加入一列 
	 */
	public void extendWithBias(double[][] pv, double bias) {
		double[] temp = null;
		int length = 0;
		int counter = 0;
		for(int i = 0; i < pv.length; i++) {
			length = pv[i].length + 1;
			temp = new double[length];
			counter = 0;
			for(int j = 0; j < pv[i].length; j++) {
				temp[counter++] = pv[i][j];
			}
			temp[counter] = bias;
			pv[i] = temp;
		}
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
	 *	交叉验证 ，返回预测micro-f1, macro-f1, hamming loss
	 */
	public double[] crossValidation(Problem prob, Parameter param, int n_fold, int k1, int k2, double c) {
		
		param.setC(c);
		
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
			
			MetaFeature mf = new MetaFeature(train);
			double[][] trainp = mf.transTrainSet(train, k1);
			
			double[][] validp = mf.transTestSet(train, valid, k1);
			
			train.x = mf.trans(trainp);
			train.n = trainp[0].length;
			
			valid.x = mf.trans(validp);
			train.n = validp[0].length;
			
			DataPoint[][] w = train(train, param);
			
			double[][] trainpv = predictValues(w, train.x, train.n);
			scale(trainpv);
			
			double[][] validpv = predictValues(w, valid.x, train.n);
			scale(validpv);
			
			int[][] predictLabel = predictNear(trainpv, validpv, train.y, k2);
			
			for(int j = 0; j < validIndex.length; j++) {
				pre[validIndex[j]] = predictLabel[j];
			}
		}
		double microf1 = Measures.microf1(this.uniqueLabels, prob.y, pre);
		double macrof1 = Measures.macrof1(this.uniqueLabels, prob.y, pre);
		double hammingloss = Measures.averageSymLoss(prob.y, pre);
		
		double[] result = {microf1, macrof1, hammingloss};
		
		return result;
	}
	
	/**
	 * 
	 */
	public double[] pbpu(Problem prob, Parameter param, int[] k1s, int[] k2s, double[] cs) {
		int k1;
		int k2;
		double c;
		
		k1 = k1s[(int)(Math.random() * k1s.length)];
		k2 = k2s[(int)(Math.random() * k2s.length)];
		c = cs[(int)(Math.random() * cs.length)];
		
		double perf = Double.POSITIVE_INFINITY;
		double[] tperf;
		
		int counter = 0;
		while(counter < 10) {
			perf = Double.POSITIVE_INFINITY;
			for(int i = 0; i < k1s.length; i++) {
				int tk1 = k1s[i];
				tperf = crossValidation(prob, param, 5, tk1, k2, c);
				
				System.out.println("k1 = " + tk1 + ", k2 = " + k2 + ", c = " + c + 
					", " + tperf[0] + ", " + tperf[1] + ", " + tperf[2]);
				
				if(tperf[2] < perf) {
					k1 = k1s[i];
					perf = tperf[2];
				}
			}
			
			perf = Double.POSITIVE_INFINITY;
			for(int i = 0; i < k2s.length; i++) {
				int tk2 = k2s[i];
				tperf = crossValidation(prob, param, 5, k1, tk2, c);
				
				System.out.println("k1 = " + k1 + ", k2 = " + tk2 + ", c = " + c + 
						", " + tperf[0] + ", " + tperf[1] + ", " + tperf[2]);
				
				if(tperf[2] < perf) {
					perf = tperf[2];
					k2 = k2s[i];
				}
			}
			
			perf = Double.POSITIVE_INFINITY;
			for(int i = 0; i < cs.length; i++) {
				double tc = cs[i];
				tperf = crossValidation(prob, param, 5, k1, k2, tc);
				
				System.out.println("k1 = " + k1 + ", k2 = " + k2 + ", c = " + tc + 
						", " + tperf[0] + ", " + tperf[1] + ", " + tperf[2]);
				
				if(tperf[2] < perf) {
					perf = tperf[2];
					c = cs[i];
				}
			}
			
			counter = counter + 1;
			System.out.println();
		}
		return null;
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
	public int[][] predictNear(double[][] pv, double[][] testpv, int[][] y, int k) {
		int[][] result = new int[testpv.length][];
		
		for(int i = 0; i < result.length; i++) {
			result[i] = getNearestLabel(pv, testpv[i], y, k);
		}
		
		return result;
	}
	
	/**
	 * 使用训练样本本身选择最佳的k
	 */
	public int selectK(double[][] pv, int[][] y, int[] k) {
		int counter = 0;
		double[] distance = new double[pv.length - 1];
		int[][] ty = new int[y.length - 1][];
		double[] sub = null;
		int[][] py = null;
		
		int[][][] predictLabels = new int[k.length][pv.length][];
		
		for(int i = 0; i < pv.length; i++) {
			counter = 0;
			for(int j = 0; j < pv.length; j++) {
				if(i != j) {
					sub = SparseVector.subVector(pv[i], pv[j]);
					distance[counter] = SparseVector.innerProduct(sub, sub);
					ty[counter] = y[j];
					counter++;
				}
			}
			
			int[] index = Sort.getIndexBeforeSort(distance);
			
			for(int m = 0; m < k.length; m++) {
				py = getFirstKY(index, ty, k[m]);
				int[] predicty = voteLabel(py);
				predictLabels[m][i] = predicty;
			}
		}
		
		double[][] performances = new double[k.length][2];
		double bestp = Double.POSITIVE_INFINITY;
		int ind = -1;
		for(int i = 0; i < k.length; i++) {
			double microf1 = Measures.microf1(this.uniqueLabels, y, predictLabels[i]);
			double macrof1 = Measures.macrof1(this.uniqueLabels, y, predictLabels[i]);
			performances[i][0] = microf1;
			performances[i][1] = macrof1;
			System.out.println("K = " + k[i] + ", Micro-F1 = " + microf1 + ", Macro-F1 = " + macrof1);
			if(microf1 >= bestp) {
				bestp = microf1;
				ind = i;
			}
		}
		
		return k[ind];
	}
	
	/**
	 * 
	 */
	public int selectKHammingLoss(double[][] pv, int[][] y, int[] k) {
		int counter = 0;
		double[] distance = new double[pv.length - 1];
		int[][] ty = new int[y.length - 1][];
		double[] sub = null;
		int[][] py = null;
		
		int[][][] predictLabels = new int[k.length][pv.length][];
		
		for(int i = 0; i < pv.length; i++) {
			counter = 0;
			for(int j = 0; j < pv.length; j++) {
				if(i != j) {
					sub = SparseVector.subVector(pv[i], pv[j]);
					distance[counter] = SparseVector.innerProduct(sub, sub);
					ty[counter] = y[j];
					counter++;
				}
			}
			
			int[] index = Sort.getIndexBeforeSort(distance);
			
			for(int m = 0; m < k.length; m++) {
				py = getFirstKY(index, ty, k[m]);
				int[] predicty = voteLabel(py);
				predictLabels[m][i] = predicty;
			}
		}
		
		double bestp = Double.POSITIVE_INFINITY;
		int ind = -1;
		for(int i = 0; i < k.length; i++) {			
			double loss = Measures.averageSymLoss(y, predictLabels[i]);
			if(loss <= bestp) {
				bestp = loss;
				ind = i;
			}
			System.out.println("K = " + k[i] + ", average hamming loss = " + loss);
		}
		
		return k[ind];
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
	 *  以多类标方式给出预测类别
	 */
	public int[] voteMultiLabel(int[][] y) {
		double n = y.length;
		double[] t = new double[(int)n];
		for(int i = 0; i < y.length; i++) {
			t[i] = 0;
			for(int j = 0; j < y.length; j++) {
				if(isSameVector(y[i], y[j])) {
					t[i] = t[i] + 1;
				}
			}
		}
		
		double max = Double.NEGATIVE_INFINITY;
		for(int i = 0; i < t.length; i++) {
			if(t[i] > max) {
				max = t[i];
			}
		}
		
		List<Integer> list = new ArrayList<Integer>();
		for(int i = 0; i < t.length; i++) {
			if(t[i] == max) {
				list.add(i);
			}
		}
		
		int id = list.get(0);
		return y[id];
	}
			
			
	/**
	 * 
	 */
	public boolean isSameVector(int[] a, int[] b) {
		if(a == null || b == null) {
			return false;
		}
		
		Set<Integer> set = new HashSet<Integer>();
		for(int i = 0; i < a.length; i++) {
			set.add(a[i]);
		}
		
		boolean flag = true;
		for(int i = 0; i < b.length; i++) {
			if(!set.contains(b[i])) {
				flag = false;
				break;
			}
		}
		return flag;
	}
	
	/**
	 * c惩罚, k近邻, n_fold交叉验证
	 */
	public double[][] gridSerach(Problem prob, Parameter param, int n_fold, double c, int[] k) {
		int n = prob.l;
		
		int[][][] pre = new int[k.length][n][];
		
		int[] index = RandomSequence.randomSequence(n);
		
		int segLength = n / n_fold;
		
		int vbegin = 0;
		int vend = 0;		
		
		param.setC(c);
		
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
			
			MetaFeature mf = new MetaFeature(train);
			double[][] trainp = mf.transTrainSet(train, 150);
			
			double[][] validp = mf.transTestSet(train, valid, 150);
			
			train.x = mf.trans(trainp);
			train.n = trainp[0].length;
			
			valid.x = mf.trans(validp);
			valid.n = validp[0].length;
					
			DataPoint[][] w = train(train, param);
			
			double[][] trainpv = predictValues(w, train.x, train.n);
			scale(trainpv);
			
			double[][] validpv = predictValues(w, valid.x, valid.n);
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
		
		double[][] performance = new double[k.length][2];
		for(int i = 0; i < k.length; i++) {
			double microf1 = Measures.microf1(this.uniqueLabels, prob.y, pre[i]);
			double macrof1 = Measures.macrof1(this.uniqueLabels, prob.y, pre[i]);
			performance[i][0] = microf1;
			performance[i][1] = macrof1;		
			double hammingLoss = Measures.averageSymLoss(prob.y, pre[i]);
			double zeroneloss = Measures.zeroOneLoss(prob.y, pre[i]);
			System.out.println("C = " + c + ", K = " + k[i] + 
					", Micro-F1 = " + microf1 + ", Macro-F1 = " + macrof1 + 
					", Hamming Loss = " + hammingLoss + 
					", zero one loss = " + zeroneloss);
		}
		return performance;
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
	 * 根据目录结构返回cost
	 */
	public double[] getCost(int[][] y, int id) {
		double maxCost = this.structure.getMaxDistance();
		double pr = positiveRatio(y, id);
		
		double[] cost = new double[y.length];
		for(int i = 0; i < y.length; i++) {
			cost[i] = getCost(y[i], id, maxCost);
//			cost[i] = getCost(y[i], id, maxCost, pr);
		}
		return cost;
	}
	
	/**
	 * 
	 */
	public double getCost(int[] y, int id, double maxCost) {
		double distance = 1;
		if(Contain.contain(y, id)) {
			distance = maxCost;
		} else {
			double max = Double.NEGATIVE_INFINITY;
			for(int i = 0; i < y.length; i++) {
				double tempcost = this.structure.getDistance(id, y[i]);
				tempcost = maxCost - tempcost;
				if(tempcost > max) {
					max = tempcost;
				}
			}
			distance = max;
		}
		return Math.max(distance, 1);
	}
	
	
	/**
	 * 
	 */
	public double getCost(int[] y, int id, double maxCost, double pr) {
		double distance = 1;
		if(Contain.contain(y, id)) {
			distance = maxCost / pr;
		} else {
			double max = Double.NEGATIVE_INFINITY;
			for(int i = 0; i < y.length; i++) {
				double tempcost = this.structure.getDistance(id, y[i]);
				tempcost = maxCost - tempcost;
				if(tempcost > max) {
					max = tempcost;
				}
			}
			distance = max * pr;
		}
		return Math.max(distance, 1);
	}
	
	/**
	 * 正例所占比例
	 */
	public double positiveRatio(int[][] y, int id) {
		double totle = y.length;
		double counter = 0;
		for(int i = 0; i < y.length; i++) {
			if(Contain.contain(y[i], id)) {
				counter++;
			}
		}
		return counter / totle;
	}
	
	/**
	 *	交叉验证 ，返回支持向量机预测输出
	 */
	public double[][] crossValidation(Problem prob, Parameter param, int n_fold, int k1, double c) {
		
		param.setC(c);
		
		int n = prob.l;
		
		double[][] pre = new double[n][];
		
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
			
			MetaFeature mf = new MetaFeature(train);
			double[][] trainp = mf.transTrainSet(train, k1);
			
			double[][] validp = mf.transTestSet(train, valid, k1);
			
			train.x = mf.trans(trainp);
			train.n = trainp[0].length;
			
			valid.x = mf.trans(validp);
			valid.n = validp[0].length;
			
			DataPoint[][] w = train(train, param);
				
			double[][] validpv = predictValues(w, valid.x, valid.n);
			scale(validpv);
			
			for(int j = 0; j < validIndex.length; j++) {
				pre[validIndex[j]] = validpv[j];
			}
		}
		return pre;
	}
	
	/**
	 * 
	 */
	public double[] getKnnLabels(double[][] pre, int[][] y, int k) {
		int[][] predict = new int[pre.length][];
		for(int i = 0; i < pre.length; i++) {
			double[] dis = new double[pre.length];
			for(int j = 0; j < pre.length; j++) {
				double[] sub = SparseVector.subVector(pre[i], pre[j]);
				dis[j] = SparseVector.innerProduct(sub, sub);
			}
			
			int[] index = Sort.getIndexBeforeSort(dis);
			int[][] kl = new int[k][];
			for(int j = 0; j < k; j++) {
				kl[j] = y[index[j+1]];
			}
			
			predict[i] = getLabel(kl);
		}
		double microf1 = Measures.microf1(this.uniqueLabels, y, predict);
		double macrof1 = Measures.macrof1(this.uniqueLabels, y, predict);
		double hammingloss = Measures.averageSymLoss(y, predict);
		
		double[] result = {microf1, macrof1, hammingloss};
		return result;
	}
	
	/**
	 * 
	 */
	public int[] getLabel(int[][] kl) {
		double n = kl.length / 2;
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		
		int key;
		int value;
		for(int i = 0; i < kl.length; i++) {
			for(int j = 0; j < kl[i].length; j++) {
				key = kl[i][j];
				if(map.containsKey(key)) {
					value = map.get(key);
					value = value + 1;
					map.put(key, value);
				} else {
					map.put(key, 1);
				}
			}
		}
		
		Set<Integer> set = map.keySet();
		Iterator<Integer> it = set.iterator();
		
		List<Integer> list = new ArrayList<Integer>();
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
	public void train(Problem prob, Parameter param, int nfold, int[] k1, double[] cs, int[] k2) {
		for(int i = 0; i < k1.length; i++) {
			for(int j = 0; j < cs.length; j++) {
				double[][] tpv = crossValidation(prob, param, 5, k1[i], cs[j]);
				for(int k = 0; k < k2.length; k++) {
					double[] perf = crossPerf(tpv, prob.y, k2[k], 5);
					System.out.println("k1=" + k1[i] + ", c=" + cs[j] + ", k2=" + k2[k]
							+ ", Micro-F1=" + perf[0] + ", Macro-F1="+perf[1] + 
							", Hamming Loss =" + perf[2]);
				}
			}
		}
	}
	
	/**
	 * 
	 */
	public int[][] getCrossKnnLabels(double[][] pre, int[][] y, int k, int n_fold) {
		double[][] trainpv = null;
		int[][]    trainLabel = null;
		double[][] validpv = null;
		
		int n = pre.length;
		int[][] predict = new int[pre.length][];
		
		int[] index = RandomSequence.randomSequence(pre.length);
		
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
			
			trainpv = new double[trainIndex.length][];
			trainLabel = new int[trainIndex.length][];
			for(int j = 0; j < trainpv.length; j++) {
				trainpv[j] = pre[trainIndex[j]];
				trainLabel[j] = y[trainIndex[j]];
			}
			
			validpv = new double[validIndex.length][];
			for(int j = 0; j < validpv.length; j++) {
				validpv[j] = pre[validIndex[j]];
			}
			
			int[][] validpre = new int[validpv.length][];
			for(int j = 0; j < validpre.length; j++) {
				validpre[j] = getNearestLabel(trainpv, validpv[j], trainLabel, k);
			}
			
			for(int j = 0; j < validpre.length; j++) {
				predict[validIndex[j]] = validpre[j];
			}
		}
		return predict;
	}
	
	/**
	 * 
	 */
	public double[] crossPerf(double[][] pre, int[][] y, int k, int n_fold) {
		int[][] predict = getCrossKnnLabels(pre, y, k, n_fold);
		double microf1 = Measures.microf1(this.uniqueLabels, y, predict);
		double macrof1 = Measures.macrof1(this.uniqueLabels, y, predict);
		double hammingloss = Measures.averageSymLoss(y, predict);
		
		double[] result = {microf1, macrof1, hammingloss};
		return result;
	}
}
