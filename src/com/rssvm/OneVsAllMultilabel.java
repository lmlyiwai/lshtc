package com.rssvm;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;
import com.structure.Structure;
import com.tools.Contain;
import com.tools.RandomSequence;
import com.tools.Sort;

public class OneVsAllMultilabel {
	private Problem 		prob;
	private Parameter 		param;
	private int[] 			uniqueLabels;
	private DataPoint[][] 	weight;
	private DataPoint[][] 	weight1;
	private Structure		structure;
	
	public OneVsAllMultilabel(Problem prob, Parameter param) {
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
//			double[] cost = getCost(prob.y, label);
//			w[i] = Linear.train(prob, y, param, null, cost, tloss, null, 0);
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
	 *	交叉验证 ，返回预测类标
	 */
	public int[][] crossValidation(Problem prob, Parameter param, int n_fold) {
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
			
			DataPoint[][] w = train(train, param);
			
			double[][] pv = predictValues(w, valid.x);
//			int[][] predictLabel = predict(pv);
			int[][] predictLabel = predictMax(pv);
			
			for(int j = 0; j < validIndex.length; j++) {
				pre[validIndex[j]] = predictLabel[j];
			}
		}
		double microf1 = Measures.microf1(this.uniqueLabels, prob.y, pre);
		double macrof1 = Measures.macrof1(this.uniqueLabels, prob.y, pre);
		double hammingloss = Measures.averageSymLoss(prob.y, pre);
		System.out.println("c = " + param.getC() + ", Micro-F1 = " + microf1 + ", Macro-F1 = " 
				+ macrof1 + ", Hamming Loss = " + hammingloss);
		return pre;
	}
	
	/**
	 *	指定C变化范围，取性能最好是所对应C，并在所有样本上训练支持向量机
	 */
	public DataPoint[][] selectModel(Problem prob, Parameter param, double[] C) {
		double[] performance = new double[C.length];
		
		double microf1 = 0;
		double macrof1 = 0;
		double c;
		int[][] pre;
		for(int i = 0; i < C.length; i++) {
			c = C[i];
			param.setC(c);
			pre = crossValidation(prob, param, 10);
			microf1 = Measures.microf1(this.uniqueLabels, prob.y, pre);
			performance[i] = microf1;
			System.out.println("C = " + c + ":" + performance[i]);
		}
		
		double max = Double.NEGATIVE_INFINITY;
		int index = -1;
		for(int i = 0; i < performance.length; i++) {
			if(performance[i] >= max) {
				index = i;
				max = performance[i];
			}
		}
		
		c = C[index];
		param.setC(c);
		
		DataPoint[][] w = train(prob, param);
		return w;
	}
	
	/**
	 *  迭代优化
	 */
	public void revisedTrain(Problem prob, Parameter param, double[] C) {
		
		DataPoint[][] w = selectModel(prob, param, C);
		
		double[][] pv = predictValues(w, prob.x);
		
		
		DataPoint[][] x = trans(pv, -1);
		
		System.out.println();
		
		prob.x = x;
		
		DataPoint[][] w1 = selectModel(prob, param, C);
		
		this.weight = w;
		this.weight1 = w1;
	}
	
	/**
	 *	预测输出 
	 */
	public int[][] revisedPredict(DataPoint[][] x) {
		double[][] pv = predictValues(this.weight, x);
		
		
		double[][] pv1 = predictValues(this.weight1, pv);
		
		int[][] pre = predict(pv1);
		
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
			double sum = SparseVector.innerProduct(sub, sub);
			distance[i] = Math.pow(sum, 0.5);
		}
		
		int[] index = Sort.getIndexBeforeSort(distance);
		
		int[][] pre = new int[n][];
		for(int i = 0; i < n; i++) {
			pre[i] = y[index[i]];
		}
		
		int[] result = sigleLabel(pre);
//		int[] result = multiClass(pre);
		return result;
	}
	

	/**
	 * y其实为一列向量
	 */
	public int[] multiClass(int[][] y) {
		int[] ys = new int[y.length];
		int[] ys_count = new int[y.length];
		int pointer = -1;
		boolean contain = false;
		
		for(int i = 0; i < y.length; i++) {
			int ty = y[i][0];
			contain = false;
			for(int j = 0; j <= pointer; j++) {
				if(ty == ys[j]) {
					ys_count[j]++;
					contain = true;
				}
			}
			
			if(!contain) {
				ys[++pointer] = ty;
				ys_count[pointer] = 1;
			}
		}
		
		int max = Integer.MIN_VALUE;
		int index = -1;
		for(int i = 0; i <= pointer; i++) {
			if(ys_count[i] > max) {
				max = ys_count[i];
				index = i;
			}
		}
		
		int[] label = new int[1];
		label[0] = ys[index];
		return label;
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
	
	public int[][] newPredictNear(double[][] pv, double[][] testpv, int[][] y, int k, double exponent) {
		int[][] result = new int[testpv.length][];
		for(int i = 0; i < result.length; i++) {
			System.out.println(i);
			result[i] = newGetNearestLabel(pv, testpv[i], y, k, exponent);
		}
		return result;
	}
	
	public int[] newGetNearestLabel(double[][] pv, double[] testpv, int[][] y, int k, double exponent) {
		double[] dis = distance(pv, testpv);
		int[] ind = Sort.getIndexBeforeSort(dis);

		int[] findex = getFirstKindex(ind, k);
		int[][] firstK = getFirstKLabels(y, findex);
		int[] labels = getUFirstLabels(firstK);
		double[] sims = getKSim(pv, testpv, findex);
		int[] tpy = getFinalLabels(y, firstK, labels, sims, exponent);
		return tpy;
	}
	
	/**
	 * 
	 */
	public int[] predictNear(double[][] pv, double[] testpv, int[][] y, int k) {
		int[] result = null;
		result = getNearestLabel(pv, testpv, y, k);
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
	public int[] sigleLabel(int[][] y) {
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
		
		Set<Integer> keySet = map.keySet();
		Iterator<Integer> it = keySet.iterator();
		int max = Integer.MIN_VALUE;
		int flabel = 0;
		while(it.hasNext()) {
			key = it.next();
			value = map.get(key);
			
			if(value > max) {
				max = value;
				flabel = key;
			}
		}
		int[] result = {flabel};
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
			System.out.println("Fold " + i);
			
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
		
		double[][] performance = new double[k.length][4];
		for(int i = 0; i < k.length; i++) {
			double microf1 = Measures.microf1(this.uniqueLabels, prob.y, pre[i]);
			double macrof1 = Measures.macrof1(this.uniqueLabels, prob.y, pre[i]);		
			double hammingLoss = Measures.averageSymLoss(prob.y, pre[i]);
			double zeroneloss = Measures.zeroOneLoss(prob.y, pre[i]);
			System.out.println("C = " + c + ", K = " + k[i] + 
					", Micro-F1 = " + microf1 + ", Macro-F1 = " + macrof1 + 
					", Hamming Loss = " + hammingLoss + 
					", zero one loss = " + zeroneloss);
			
			performance[i][0] = microf1;
			performance[i][1] = macrof1;
			performance[i][2] = hammingLoss;
			performance[i][3] = zeroneloss;
		}
		 
		return performance;
	}
	
	//KNN处理类别不均衡问题
	public double[][] newGridSerach(Problem prob, Parameter param, int n_fold, double c, int[] k,
			double exponent) {
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
			
			DataPoint[][] w = train(train, param);
			
			double[][] trainpv = predictValues(w, train.x);
			scale(trainpv);
			
			double[][] validpv = predictValues(w, valid.x);
			scale(validpv);

			int[][][] temppre = new int[k.length][valid.l][];
			for(int h = 0; h < validpv.length; h++) {
				double[] dis = distance(trainpv, validpv[h]);
				int[] ind = Sort.getIndexBeforeSort(dis);
				
				
				for(int m = 0; m < k.length; m++) {
					int[] findex = getFirstKindex(ind, k[m]);
					int[][] firstK = getFirstKLabels(train.y, findex);
					int[] labels = getUFirstLabels(firstK);
					double[] sims = getKSim(trainpv, validpv[h], findex);
					int[] tpy = getFinalLabels(train.y, firstK, labels, sims, exponent);
					temppre[m][h] = tpy;
				}
			}
			
			for(int h = 0; h < valid.l; h++) {
				for(int m = 0; m < k.length; m++) {
					pre[m][validIndex[h]] = temppre[m][h];
				}
			}
		}
		
		double[][] performance = new double[k.length][4];
		for(int i = 0; i < k.length; i++) {
			double microf1 = Measures.microf1(this.uniqueLabels, prob.y, pre[i]);
			double macrof1 = Measures.macrof1(this.uniqueLabels, prob.y, pre[i]);		
			double hammingLoss = Measures.averageSymLoss(prob.y, pre[i]);
			double zeroneloss = Measures.zeroOneLoss(prob.y, pre[i]);
			System.out.println("C = " + c + ", K = " + k[i] + 
					", Micro-F1 = " + microf1 + ", Macro-F1 = " + macrof1 + 
					", Hamming Loss = " + hammingLoss + 
					", zero one loss = " + zeroneloss);
			
			performance[i][0] = microf1;
			performance[i][1] = macrof1;
			performance[i][2] = hammingLoss;
			performance[i][3] = zeroneloss;
		}
		 
		return performance;
	}
	
	public int[][] getFirstKLabels(int[][] y, int[] index) {
		int[][] labels = new int[index.length][];
		for(int i = 0; i < index.length; i++) {
			labels[i] = y[index[i]];
		}
		return labels;
	}
	
	public int[] getUFirstLabels(int[][] labels) {
		Set<Integer> set = new HashSet<Integer>();
		for(int i = 0; i < labels.length; i++) {
			for(int j = 0; j < labels[i].length; j++) {
				set.add(labels[i][j]);
			}
		}
		
		int[] result = new int[set.size()];
		int count = 0;
		Iterator<Integer> it = set.iterator();
		while(it.hasNext()) {
			result[count] = it.next();
			count++;
		}
		return result;
	}
	
	public double[] getTrainNum(int[][] y, int[] labels) {
		double[] result = new double[labels.length];
		for(int i = 0; i < labels.length; i++) {
			for(int j = 0; j < y.length; j++) {
				if(Contain.contain(y[j], labels[i])) {
					result[i]++;
				}
			}
		}
		return result;
	}
	
	public double[] getKSim(double[][] train, double[] test, int[] index) {
		double[] sims = new double[index.length];
		for(int i = 0; i < sims.length; i++) {
			sims[i] = sim(train[index[i]], test);
		}
		return sims;
	}
	
	public boolean ifContainLabel(int[][] y, int[][] firstK, int label, double[] sims, double exponent) {
		double positive = 0;
		double negative = 0;
		
		int[] labels = {label};
		double[] pon = getTrainNum(y, labels);
		double ponn = pon[0];
		double negg = y.length - ponn;
		double wp = 1.0 / (Math.pow(ponn / Math.min(ponn, negg), 1.0 / exponent));
		double wn = 1.0 / (Math.pow(negg / Math.min(ponn, negg), 1.0 / exponent));
		for(int i = 0; i < sims.length; i++) {
			if(Contain.contain(firstK[i], label)) {
				positive += wp * sims[i];
			} else {
				negative += wn * sims[i];
			}
		}
		if(positive >= negative) {
			return true;
		} else {
			return false;
		}
	}
	
	public int[] getFinalLabels(int[][] y, int[][] firstK, int[] labels, double[] sims, double exponent) {
		List<Integer> list = new ArrayList<Integer>();
		for(int i = 0; i < labels.length; i++) {
			boolean flag = ifContainLabel(y, firstK, labels[i], sims, exponent);
			if(flag == true) {
				list.add(labels[i]);
			}
		}
		int[] result = new int[list.size()];
		for(int i = 0; i < list.size(); i++) {
			result[i] = list.get(i);
		}
		return result;
	}
	
	public int[] getFirstKindex(int[] index, int k) {
		int[] result = new int[k];
		for(int i = 0; i < k; i++) {
			result[i] = index[i];
		}
		return result;
	}
	
	
	public double sim(double[] va, double[] vb) {
		double inpab = SparseVector.innerProduct(va, vb);
		double inpaa = SparseVector.innerProduct(va, va);
		Math.pow(inpaa, 0.5);
		double inpbb = SparseVector.innerProduct(vb, vb);
		Math.pow(inpbb, 0.5);
		return (inpab / (inpaa * inpbb));
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
	public double[] l1Distance(double[][] pv, double[] p) {
		double[] result = new double[pv.length];
		double[] sub = null;
		double sum = 0;
		for(int i = 0; i < result.length; i++) {
			sub = SparseVector.subVector(pv[i], p);
			sum = 0;
			for(int j = 0; j < sub.length; j++) {
				sum += Math.abs(sub[j]);
			}
			result[i] = sum;
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
	 * 
	 */
	public int getBestHammingLoss(double[][] perf, int[] k) {
		double min = Double.POSITIVE_INFINITY;
		int index = -1;
		for(int i = 0; i < perf.length; i++) {
			if(perf[i][2] < min) {
				min = perf[i][2];
				index = i;
			}
		}
		return k[index];
	}
}
