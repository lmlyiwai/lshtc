package com.stack;

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

public class BinarySVMKnn {
	private Problem 		prob;
	private Parameter 		param;
	private int[] 			uniqueLabels;
	private DataPoint[][] 	weight;
	private Structure		structure;
	private double[]        uks;
	private double[][]		ks;
	private double[][]		kc;
	
	public BinarySVMKnn(Problem prob, Parameter param) {
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
			param.setC(this.kc[i][0]);
			w[i] = Linear.train(prob, y, param, null, tloss, null, 0);
		}
		return w;
	} 
	
	/**
	 *	���ѵ�����г��ֵı�ǩ 
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
	 *	���idָ�����ǩ 
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
	 *	 Ԥ�����ֵ
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
	 * 
	 */
	public double[] predictValues(DataPoint[] w, DataPoint[][] x) {
		double[] pv = new double[x.length];
		DataPoint[] tx;
		for(int j = 0; j < x.length; j++) {
			tx = x[j];
			pv[j] = SparseVector.innerProduct(w, tx); 
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
	 *	Ԥ����� 
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
	 * �����Ԥ�� 
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
	 *	ȫ����ת��ΪDataPoint��ʽ 
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
	 *	����һ�� 
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
	 *	������һ�� 
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
	public void allCrossValidation(Problem prob, Parameter param, int n_fold, int[] k, double[] c) {
		double[][] kc = new double[this.uniqueLabels.length][2];
		int label = 0;
		for(int i = 0; i < this.uniqueLabels.length; i++) {
			label = this.uniqueLabels[i];
			kc[i] = binaryCrossValidation(prob, param, n_fold, k, c, label);
		}
		this.kc = kc;
	}
	
	/**
	 * �������k,c
	 */
	public double[] binaryCrossValidation(Problem prob, Parameter param, int n_fold, int[] k, double[] c, int label) {
		double[][] perf = new double[c.length][k.length];
		
		double bestPerf = Double.NEGATIVE_INFINITY;
		int row = 0;
		int col = 0;
		for(int i = 0; i < c.length; i++) {
			double tc = c[i];
			param.setC(tc);
			for(int j = 0; j < k.length; j++) {
				int tk = k[j];
				perf[i][j] = crossValidation(prob, param, n_fold, tk, label);
				if(perf[i][j] > bestPerf) {
					bestPerf = perf[i][j];
					row = i;
					col = j;
				}
			}
		}
		
		double[] result = new double[2];
		result[0] = c[row];
		result[1] = k[col];
		
		System.out.println("label = " + label + ", c = " + result[0] + ", k = " + result[1] +
				", perf = " + bestPerf);
		return result;
	}
	
	/**
	 *	������֤ ������Ԥ�����
	 */
	public double crossValidation(Problem prob, Parameter param, int n_fold, int k, int label) {
		int[] labels = getLabels(prob.y, label);
		
		int n = prob.l;
		
		int[] pre = new int[n];
		
		int[] index = RandomSequence.randomSequence(n);
		
		int segLength = n / n_fold;
		
		int vbegin = 0;
		int vend = 0;		
		
		int[] validIndex = null;
		int[] trainIndex = null;
		int counter = 0;
		
		int[] labely = null;
		
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
			train.y = new int[trainIndex.length][1];
			
			labely = new int[train.l];
			
			counter = 0;
			for(int j = 0; j < trainIndex.length; j++) {
				train.x[counter] = prob.x[trainIndex[j]];
				train.y[counter][0] = labels[trainIndex[j]];
				labely[counter] = labels[trainIndex[j]];
				counter++;
			}
			
			Problem valid = new Problem();
			valid.l = validIndex.length;
			valid.n = prob.n;
			valid.bias = prob.bias;
			valid.x = new DataPoint[validIndex.length][];
			valid.y = new int[validIndex.length][1];
			
			counter = 0;
			for(int j = 0; j < validIndex.length; j++) {
				valid.x[counter] = prob.x[validIndex[j]];
				valid.y[counter][0] = labels[validIndex[j]];
				counter++;
			}
			
			double[] tloss = new double[1];
			DataPoint[] w = Linear.train(train, labely, param, null, tloss, null, 0);
			
			double[] trainpv = predictValues(w, train.x);
			
			double[] validpv = predictValues(w, valid.x);
			
			int[] validPre = getknnLabels(trainpv, validpv, labely, k);
			
			for(int j = 0; j < validIndex.length; j++) {
				pre[validIndex[j]] = validPre[j];
			}
		}
		
		double performance = 0;
		performance = accuracy(labels, pre);
		return performance;
	}
	
	/**
	 * 
	 */
	public double accuracy(int[] tl, int[] pl) {
		double counter = 0;
		for(int i = 0; i < tl.length; i++) {
			if(tl[i] == pl[i]) {
				counter = counter + 1;
			}
		}
		
		double performance = counter / tl.length;
		return performance;
	}
	
	/**
	 * 
	 */
	public int[] getknnLabels(double[] trainpv, double[] validpv, int[] trainvy, int k) {	
		int[] pre = new int[validpv.length];
		double vpv = 0;
		for(int i = 0; i < validpv.length; i++) {
			vpv = validpv[i];
			double[] distance = new double[trainpv.length];
			for(int j = 0; j < trainpv.length; j++) {
				distance[j] = Math.abs(trainpv[j] - vpv);
			}
			
			int[] ind = Sort.getIndexBeforeSort(distance);
			pre[i] = kLabel(trainvy, ind, k);
		}
		return pre;
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
	 * c�ͷ�, k����, n_fold������֤
	 */
	public double gridSerach(Problem prob, Parameter param, int n_fold, double c, int[] k) {
		int n = prob.l;
		
		int[][] pre = new int[n][];
		
		int[] index = RandomSequence.randomSequence(n);
		
		int segLength = n / n_fold;
		
		int vbegin = 0;
		int vend = 0;		
		
		param.setC(c);
		
		int[] validIndex = null;
		int[] trainIndex = null;
		int counter = 0;
		
		double[][] 	ks = new double[this.uniqueLabels.length][n_fold];
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
			
			DataPoint[][] w = train(train, param);
			
			double[][] trainpv = predictValues(w, train.x);
			scale(trainpv);
			
			double[][] validpv = predictValues(w, valid.x);
			scale(validpv);
			
			int[][] tpre = new int[this.uniqueLabels.length][valid.l];
			for(int h = 0; h < this.uniqueLabels.length; h++) {
				int[][] py = getBinaryPredict(trainpv, validpv, train.y, k, h);
				int[] validy = getLabels(valid.y, this.uniqueLabels[h]);
				Map<String, Object> map = maxFmeature(py, validy);
				int[] p = (int[])map.get("label");
				tpre[h] = p;
				double tk = k[(int)map.get("k")];
				ks[h][i] = tk;
			}
					
			int[][] predictValid = getPre(tpre);
			
			for(int h = 0; h < valid.l; h++) {
				pre[validIndex[h]] = predictValid[h];
			}
		}
		
		this.uks = averageK(ks);
		this.ks  = ks;
		
		double performance;
		double microf1 = Measures.microf1(this.uniqueLabels, prob.y, pre);
		double macrof1 = Measures.macrof1(this.uniqueLabels, prob.y, pre);
		double hammingloss = Measures.averageSymLoss(prob.y, pre);
		
		performance = hammingloss;
		System.out.println("c = " + c + ", Micro-F1 = " + microf1 + ", Macro-F1 = "
				+ macrof1 + ", Hamming Loss = " + hammingloss);
		return performance;
	}
	
	/**
	 *  
	 */
	public int[][] getBinaryPredict(double[][] train, double[][] test, int[][] y, int[] k, int column) {
		int label = this.uniqueLabels[column];
		int[] ty = getLabels(y, label);
		
		int[][] pre = new int[test.length][k.length];
		
		double[] traincolumn = getMatrixColumn(train, column);
		double[] testcolumn = getMatrixColumn(test, column);
		
		for(int i = 0; i < testcolumn.length; i++) {
			double tv = testcolumn[i];
			int[] ind = getSortIndex(traincolumn, tv);
			int[] tyi = kLabels(ty, ind, k);
			for(int j = 0; j < tyi.length; j++) {
				pre[i][j] = tyi[j];
			}
		}
		return pre;
	}

	/**
	 * 
	 */
	public int[] kLabels(int[] ty, int[] ind, int[] k) {
		int[] result = new int[k.length];
		for(int i = 0; i < k.length; i++) {
			int tk = k[i];
			result[i] = kLabel(ty, ind, tk);
		}
		return result;
	}
	
	/**
	 * 
	 */
	public int kLabel(int[] ty, int[] ind, int k) {
		double tk = (double)k;
		double counter = 0;
		for(int i = 0; i < k; i++) {
			if(ty[ind[i]] == 1) {
				counter++;
			}
		}
		
		if(counter >= (k / 2)) {
			return 1;
		} else {
			return -1;
		}
	}
	
	/**
	 *   ����֮����±�ֵ
	 */
	public int[] getSortIndex(double[] array, double value) {
		double[] t = new double[array.length];
		for(int i = 0; i < t.length; i++) {
			t[i] = array[i] - value;
			t[i] = Math.abs(t[i]);
		}
		
		int[] ind = Sort.getIndexBeforeSort(t);
		return ind;
	}
	
	/**
	 * ��þ���һ�� 
	 */
	public double[] getMatrixColumn(double[][] matrix, int column) {
		double[] result = new double[matrix.length];
		for(int i = 0; i < result.length; i++) {
			result[i] = matrix[i][column];
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
	public Map<String, Object> maxFmeature(int[][] py, int[] tl) {
		int[] result = new int[py.length];
		int[] pre = null;
		double maxf = Double.NEGATIVE_INFINITY;
		int index = 0;
		for(int i = 0; i < py[0].length; i++) {
			pre = getMatrixColumn(py, i);
			double f = precision(tl, pre);
			if(f > maxf) {
				maxf = f;
				index = i;
			}
		}
			
		result = getMatrixColumn(py, index);
		
		Map<String, Object> map = new HashMap<String, Object>();
		map.put("label", result);
		map.put("k", index);
		return map;
	}
	
	/**
	 * 
	 */
	public double precision(int[] tl, int[] pl) {
		double counter = 0;
		for(int i = 0; i < tl.length; i++) {
			if(tl[i] == pl[i]) {
				counter = counter + 1;
			}
		}
		
		double prec = counter / tl.length;
		return prec;
	}
	
	/**
	 * 
	 */
	public int[] getMatrixColumn(int[][] matrix, int column) {
		int[] result = new int[matrix.length];
		for(int i = 0; i < result.length; i++) {
			result[i] = matrix[i][column];
		}
		return result;
	}
	
	/**
	 * 
	 */
	public int[][] getPre(int[][] pre) {
		int[][] result = new int[pre[0].length][];
		for(int i = 0; i < pre[0].length; i++) {
			int[] tcol = getMatrixColumn(pre, i);
			int counter = 0;
			for(int j = 0; j < tcol.length; j++) {
				if(tcol[j] == 1) {
					counter++;
				}
			}
			
			int[] ty = new int[counter];
			counter = 0;
			for(int j = 0; j < tcol.length; j++) {
				if(tcol[j] == 1) {
					ty[counter++] = this.uniqueLabels[j];
				}
			}
			
			result[i] = ty;
		}
		return result;
	}
	
	/**
	 * 
	 */
	public int[][] getPredictLabels(Problem train, Problem test, DataPoint[][] w) {
		double[][] trainpv = predictValues(w, train.x);
	
		double[][] testpv = predictValues(w, test.x);
		
		int[][] pre = new int[this.uniqueLabels.length][];
		for(int i = 0; i < this.uniqueLabels.length; i++) {
			pre[i] = getColumnPredict(trainpv, testpv, train.y, (int)this.kc[i][1], i);
		}
		
		int[][] result = getPre(pre);
		return result;
	}
	
	/**
	 * 
	 */
	public int[] getColumnPredict(double[][] trainpv, double[][] testpv, int[][] y, int k, int column) {
		int[] label = getLabels(y, this.uniqueLabels[column]);
		double[] traincol = getMatrixColumn(trainpv, column);
		double[] testcol = getMatrixColumn(testpv, column);
		
		int[] result = new int[testpv.length];
		double tv;
		for(int i = 0; i < result.length; i++) {
			tv = testcol[i];
			int[] ind = getSortIndex(traincol, tv);
			result[i] = kLabel(label, ind, k);
		}
		return result;
	}
	
	/**
	 * 
	 */
	public double[] averageK(double[][] ks) {
		double sum = 0;
		double[] result = new double[ks.length];
		for(int i = 0; i < result.length; i++) {
			sum = 0;
			for(int j = 0; j < ks[i].length; j++) {
				sum = sum + ks[i][j];
			}
			result[i] = Math.round((sum / ks[i].length));
//			result[i] = Math.max(result[i], 1);
		}
		return result;
	}

	public double[] getUks() {
		return uks;
	}

	public void setUks(double[] uks) {
		this.uks = uks;
	}
	
	public double[][] meanStd(double[][] ks) {
		double[][] result = new double[ks.length][2];
		double sum = 0;
		for(int i = 0; i < ks.length; i++) {
			sum = 0;
			for(int j = 0; j < ks[i].length; j++) {
				sum = sum + ks[i][j];
			}
			double mean = sum / ks[i].length;
			result[i][0] = mean;
			
			sum  = 0;
			for(int j = 0; j < ks[i].length; j++) {
				sum = sum + (ks[i][j] - mean) * (ks[i][j] - mean);
			}
			double std = Math.pow(sum / ks[i].length, 0.5);
			result[i][1] = std;
		}
		return result;
	}


	public double[][] getKs() {
		return ks;
	}


	public void setKs(double[][] ks) {
		this.ks = ks;
	}


	public double[][] getKc() {
		return kc;
	}


	public void setKc(double[][] kc) {
		this.kc = kc;
	}
	
}
