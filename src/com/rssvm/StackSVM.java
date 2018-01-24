package com.rssvm;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import javax.xml.crypto.Data;

import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;
import com.structure.Structure;
import com.tools.Contain;
import com.tools.Kernel;
import com.tools.RandomSequence;
import com.tools.Sigmoid;
import com.tools.Sort;

public class StackSVM {
	private Problem 		prob;
	private Parameter 		param;
	private int[] 			uniqueLabels;
	private DataPoint[][] 	weight;
	private DataPoint[][] 	weight1;
	private double 			threshold;
	
	public StackSVM(Problem prob, Parameter param) {
		this.prob = prob;
		this.param = param;
		this.uniqueLabels = getAllLabels(prob.y);
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
	 *	全矩阵转换为DataPoint形式 
	 */
	public DataPoint[][] trans(double[][] pv, int label, double threshold) {
		DataPoint[][] result = new DataPoint[pv.length][];
		int index1 = findIndex(this.uniqueLabels, label);
		
		int length = pv[0].length;
		
		int counter = 0;
		int[] indexs = null;
		
		int index;
		double value;
		
		for(int i = 0; i < pv.length; i++) {
			indexs = findIndex(pv[i], threshold);
			if(Contain.contain(indexs, index1)) {
				Arrays.sort(indexs);
			} else {
				int[] nind = new int[indexs.length + 1]; 
				nind[0] = index1;
				for(int j = 1; j < nind.length; j++) {
					nind[j] = indexs[j - 1];
				}
				indexs = nind;
				Arrays.sort(indexs);
			}
			
			result[i] = new DataPoint[indexs.length + 1];
			counter = 0;
			for(int j = 0; j < indexs.length; j++) { 
				index = indexs[j] + 1;
				value = pv[i][indexs[j]];
				result[i][counter++] = new DataPoint(index, value);
			}
			result[i][counter] = new DataPoint(length + 1, 1);
		}
		return result;
	}

	/**
	 * 
	 */
	public int findIndex(int[] labels, int label) {
		int index = -1;
		for(int i = 0; i < labels.length; i++) {
			if(labels[i] == label) {
				index = i;
			}
		}
		return index;
	}
	
	/**
	 * 
	 */
	public int[] getVectorIndex(double[] pv, double threshold) {
		int counter = 0;
		for(int i = 0; i < pv.length; i++) {
			if(pv[i] > threshold) {
				counter++;
			}
		}
		
		int[] rindex = new int[counter];
		counter = 0;
		for(int i = 0; i < pv.length; i++) {
			if(pv[i] > threshold) {
				rindex[counter++] = i;
			}
		}
		return rindex;
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
	
	/**
	 *	全矩阵转换为DataPoint形式 
	 */
	public DataPoint[][] trans(double[][] pv, Structure tree, int label) {
		DataPoint[][] result = new DataPoint[pv.length][];
		
		int[] path = tree.getPathToRoot(label);
		Set<Integer> set = new HashSet<Integer>();
		
		for(int i = 0; i < path.length; i++) {
			if(path[i] != tree.getRoot()) {
				set.add(path[i]);
			}
		}
		
		int counter = set.size();
		
		int[] ids = new int[counter];
		counter = 0;
		for(int i = 0; i < this.uniqueLabels.length; i++) {
			if(set.contains(this.uniqueLabels[i])) {
				ids[counter++] = i;
			}
		}
		
		int index;
		double value;
		for(int i = 0; i < pv.length; i++) {
			result[i] = new DataPoint[ids.length + 1];
			counter = 0;
			for(int j = 0; j < ids.length; j++) {
				index = ids[j] + 1;
				value = pv[i][index - 1];
				result[i][counter++] = new DataPoint(index, value); 
			}
			
			result[i][counter] = new DataPoint(pv[0].length + 1, 1);
		}
		return result;
	}
	
	/**
	 *	全矩阵转换为DataPoint形式 
	 */
	public DataPoint[][] trans(int[][] pv, double bias) {
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
			int[][] predictLabel = predict(pv);
			
			for(int j = 0; j < validIndex.length; j++) {
				pre[validIndex[j]] = predictLabel[j];
			}
		}
		return pre;
	}
	
	/**
	 *	指定C变化范围，取性能最好是所对应C，并在所有样本上训练支持向量机
	 */
	public DataPoint[][] selectModel(Problem prob, Parameter param, double[] C) {
		double[] performance = new double[C.length];
		
		double microf1 = 0;
		double macrof1 = 0;
		double hammingLoss = 0;
		
		double c;
		int[][] pre;
		for(int i = 0; i < C.length; i++) {
			c = C[i];
			param.setC(c);
			pre = crossValidation(prob, param, 5);
			microf1 = Measures.microf1(this.uniqueLabels, prob.y, pre);
			macrof1 = Measures.macrof1(this.uniqueLabels, prob.y, pre);
			hammingLoss = Measures.averageSymLoss(prob.y, pre);
			performance[i] = microf1;
			System.out.println("c = " + c + ", Micro-F1 = " + microf1 + 
					", Macro-F1 = " + macrof1 + ", HammingLoss = " + hammingLoss);
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
//		Sigmoid.tanhx(pv, 4);
//		Sigmoid.sigmoid(pv, 1);
		pv = Kernel.map(pv);
		
		DataPoint[][] x = trans(pv, -1);
		
		System.out.println();
		
		prob.x = x;
		
		DataPoint[][] w1 = selectModel(prob, param, C);
		
		this.weight = w;
		this.weight1 = w1;
	}
	
	/**
	 *  迭代优化
	 */
	public void revisedTrain(Problem prob, Parameter param1, Parameter param2, Structure tree) {
		
		DataPoint[][] w1 = new DataPoint[this.uniqueLabels.length][];
		System.out.println("Layer 1");
		for(int i = 0; i < this.uniqueLabels.length; i++) {
			
			long start = System.currentTimeMillis();
			int label = this.uniqueLabels[i];
			System.out.print(label);
			int[] y = getLabels(prob.y, label);
			double[] loss = new double[1];
			w1[i] = Linear.train(prob, y, param1, null, loss, null, 0);
			long end = System.currentTimeMillis();
			System.out.println(", time " + (end - start) + "ms");
		}
		
		
		double[][] pv = predictValues(w1, prob.x);
		
		DataPoint[][] weight2 = new DataPoint[this.uniqueLabels.length][];
		System.out.println("Layers 2");
		
		Problem np = null;
		
		for(int i = 0; i < this.uniqueLabels.length; i++) {
			np = new Problem();
			long start = System.currentTimeMillis();
			int label = this.uniqueLabels[i];
			System.out.print(label);
			int[] y = getLabels(prob.y, label);
			double[] loss = new double[1];
			DataPoint[][] x = trans(pv, tree, label);     //从当前节点至根的所有节点
			np.x = x;
			np.bias = 1;
			np.n = this.uniqueLabels.length + 1;
			np.y = prob.y;
			np.l = x.length;
			
			weight2[i] = Linear.train(np, y, param2, null, loss, null, 0);
			long end = System.currentTimeMillis();
			System.out.println(", time " + (end - start) + "ms");
		}
		
		this.weight = w1;
		this.weight1 = weight2;
	}
	
	/**
	 * 
	 */
	public int[][] getPLabels(double[][] pv) {
		int[][] pl = new int[pv.length][];
		int counter = 0;
		for(int i = 0; i < pv.length; i++) {
			counter = 0;
			for(int j = 0; j < pv[i].length; j++) {
				if(pv[i][j] > 0) {
					counter++;
				}
			}
			
			pl[i] = new int[counter];
			counter = 0;
			for(int j = 0; j < pv[i].length; j++) {
				if(pv[i][j] > 0) {
					pl[i][counter++] = this.uniqueLabels[j];
				}
			}
		}
		return pl;
	}
	
	/**
	 * 
	 */
	public int[][] predict(DataPoint[][] x, Structure tree) {
		double[][] pv = predictValues(this.weight, x);
		
		double[][] pv1 = new double[this.uniqueLabels.length][x.length];
		for(int i = 0; i < this.uniqueLabels.length; i++) {
			int label = this.uniqueLabels[i];
			DataPoint[][] tx = trans(pv, tree, label);
			System.out.println("weight " + i);
			pv1[i] = predictValues(this.weight1[i], tx);
		}
		
		int[][] pl = predictLabel(pv1);
		return pl;
	}
	
	/**
	 * 
	 */
	public int[][]  predictLabel(double[][] pv) {
		int[][] pl = new int[pv[0].length][];
		int counter = 0;
		for(int i = 0; i < pv[0].length; i++) {
			if(i % 1000 == 0) {
				System.out.println("predict " + i);
			}
			
			counter = 0;
			for(int j = 0; j < pv.length; j++) {
				if(pv[j][i] > 0) {
					counter++;
				}
			}
			
			pl[i] = new int[counter];
			counter = 0;
			for(int j = 0; j < pv.length; j++) {
				if(pv[j][i] > 0) {
					pl[i][counter++] = this.uniqueLabels[j];
				}
			}
		}
		return pl;
	}
	
	/**
	 * 
	 */
	public double[] predictValues(DataPoint[] w, DataPoint[][] tx) {
		double[] pv = new double[tx.length];
		double[] wv = SparseVector.sparseVectorToArray(w, this.uniqueLabels.length + 1);
		
		for(int i = 0; i < tx.length; i++) {
			pv[i] = SparseVector.innerProduct(wv, tx[i]);
		}
		return pv;
	}
	
	/**
	 * 第一次训练后得到的类标作为第二次的输入
	 */
	public void rTrain(Problem prob, Parameter param, double[] C) {
		
		DataPoint[][] w = selectModel(prob, param, C);
		
		double[][] pv = predictValues(w, prob.x);
		cut(pv);
		
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
//		Sigmoid.tanhx(pv, 4);
//		Sigmoid.sigmoid(pv, 1);
		pv = Kernel.map(pv);
		
		double[][] pv1 = predictValues(this.weight1, pv);
		
		int[][] pre = predict(pv1);
		
		return pre;
	}
	
	/**
	 *	预测输出 
	 */
	public int[][] rPredict(DataPoint[][] x) {
		double[][] pv = predictValues(this.weight, x);
		cut(pv);
		
		double[][] pv1 = predictValues(this.weight1, pv);
		
		int[][] pre = predict(pv1);
		
		return pre;
	}
	
	/**
	 * pv中大于0的值替换为1，小于0的值替换为-1
	 */
	public void cut(double[][] pv) {
		if(pv == null) {
			return;
		}
		
		for(int i = 0; i < pv.length; i++) {
			for(int j = 0; j < pv[i].length; j++) {
				if(pv[i][j] > 0) {
					pv[i][j] = 1;
				} else {
					pv[i][j] = -1;
				}
			}
		}
	}
	
	/**
	 * 
	 */
	public double meanNonZero(double[][] pv, double threshold) {
		double num = 0;
		for(int i = 0; i < pv.length; i++) {
			for(int j = 0; j < pv[i].length; j++) {
				if(pv[i][j] > threshold) {
					num++;
				}
			}
		}
		
		return num / pv.length;
	}
	
	/**
	 * 
	 */
	public int[] findIndex(double[] pv, double threshold) {
		int counter = 0;
		for(int i = 0; i < pv.length; i++) {
			if(pv[i] > threshold) {
				counter++;
			}
		}
		
		int[] result = new int[counter];
		counter = 0;
		for(int i = 0; i < pv.length; i++) {
			if(pv[i] > threshold) {
				result[counter++] = i;
			}
		}
		return result;
	}
	
	/**
	 * 
	 */
	public void norm(double[][] pv) {
		for(int i = 0; i < pv.length; i++) {
			double inp = SparseVector.innerProduct(pv[i], pv[i]);
			inp = Math.pow(inp, 0.5);
			for(int j = 0; j < pv[i].length; j++) {
				pv[i][j] /= inp;
			}
		}
	}
	
	/**
	 * 
	 */
	public DataPoint[][] trans(double[][] pv, int dim) {

		DataPoint[][] x = new DataPoint[pv.length][];
		
		for(int i = 0; i < pv.length; i++) {
			int[] index = Sort.getIndexBeforeSort(pv[i]);
			x[i] = new DataPoint[dim + 1];
			int counter = 0;
			for(int j = index.length - 1; j >= index.length - dim; j--) {
				x[i][counter++] = new DataPoint(index[j] + 1, pv[i][index[j]]);
			}	
			x[i][counter] = new DataPoint(pv[0].length + 1, 1.0);
		}
		return x;
	}
}
